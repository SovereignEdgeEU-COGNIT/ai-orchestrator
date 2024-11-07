import mlflow
import mlflow.pytorch
import os
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
from data import load_data, scale_data, create_sequences
from myLSTM import LSTMModel, train_lstm_model
from myFFNN import FFNNModel, train_ffnn_model
from myGRU import GRUModel, train_gru_model
from myTCN import TCNModel, train_tcn_model
import random
import argparse

from rmse import calculate_rmse

# Define the models directory
models_uni_dir = 'models_uni'
if not os.path.exists(models_uni_dir):
    os.makedirs(models_uni_dir)
    print(f"Created directory: {models_uni_dir}")

# Hyperparameters
model_type = 'FFNN'  # Change this to 'LSTM', 'FFNN', 'GRU', or 'TCN'
input_size = 1  # Set to 1 for univariate
hidden_size = 64
num_layers = 2  # Only used for LSTM and GRU
output_size = 1  # Predicting one feature
sequence_length = 99
prediction_length = 1
batch_size = 64
learning_rate = 0.001
num_epochs = 10
train_test_ratio = 0.8

# Start MLflow experiment
mlflow.set_experiment("Univariate Model Comparison")

# Load and preprocess data
print("Data preparation started...")
filename = 'workload_series.npy'
data_reshaped = load_data(filename)
data_scaled, scaler = scale_data(data_reshaped)

# Define metrics/features for univariate prediction
metrics = ['CPU', 'Memory', 'Disk Write', 'Network Received']

# Create a directory for models
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Loop over each metric
for feature_index, metric in enumerate(metrics):
    print(f"\nTraining model for {metric}...")

    # Extract a single feature for univariate prediction
    y = data_scaled[:, feature_index].reshape(-1, 1)  # Ensure y is 2D
    X, y = create_sequences(y, sequence_length, prediction_length)

    # Split data
    train_size = int(len(X) * train_test_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    print("X_train_tensor shape:", X_train_tensor.shape)
    print("y_train_tensor shape:", y_train_tensor.shape)

    # Create dataset and dataloader
    train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Start an MLflow run
    with mlflow.start_run(run_name=f"{metric} {model_type} Model Run"):
        # Log hyperparameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers if model_type in ['LSTM', 'GRU'] else 'N/A')
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)

        # Initialize model
        if model_type == 'LSTM':
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        elif model_type == 'FFNN':
            model = FFNNModel(input_size * sequence_length, hidden_size, output_size)
        elif model_type == 'GRU':
            model = GRUModel(input_size, hidden_size, num_layers, output_size)
        elif model_type == 'TCN':
            tcn_channels = [64, 64, 64]
            model = TCNModel(input_size, output_size, tcn_channels, kernel_size=2, dropout=0.2)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training function
        train_func_map = {
            'LSTM': train_lstm_model,
            'FFNN': train_ffnn_model,
            'GRU': train_gru_model,
            'TCN': train_tcn_model
        }

        print(f"Training {model_type} model for {metric}...")
        train_func_map[model_type](model, train_loader, criterion, optimizer, num_epochs)

        # Save the model
        model_save_path = os.path.join(models_uni_dir, f'{model_type.lower()}_uni_model_{metric.lower()}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

        # Evaluation
        model.eval()
        X_test_tensor = torch.from_numpy(X_test).float()
        predictions = []
        ground_truth = []
        total_prediction_time = 0

        # Predict and measure time
        for idx in range(100):
            with torch.no_grad():
                seq = X_test_tensor[idx].unsqueeze(0)
                start_time = time.time()

                if model_type == 'FFNN':
                    seq = seq.view(seq.size(0), -1)

                pred = model(seq).cpu().numpy()
                end_time = time.time()
                total_prediction_time += (end_time - start_time)

                predictions.append(pred[0])
                ground_truth.append(y_test[idx])

        avg_prediction_time = total_prediction_time / 100
        mlflow.log_metric('avg_prediction_time_per_instance', avg_prediction_time)

        # Rescale the predictions and ground truth back to the original scale
        predictions_rescaled = np.zeros((len(predictions), 4))
        ground_truth_rescaled = np.zeros((len(ground_truth), 4))

        # Assign the predicted and true values for the current feature (column)
        predictions_rescaled[:, feature_index] = np.array(predictions).reshape(-1)
        ground_truth_rescaled[:, feature_index] = np.array(ground_truth).reshape(-1)

        # Perform inverse scaling
        predictions_rescaled = scaler.inverse_transform(predictions_rescaled)
        ground_truth_rescaled = scaler.inverse_transform(ground_truth_rescaled)

        rmse = calculate_rmse(predictions_rescaled, ground_truth_rescaled)
        mlflow.log_metric(f'RMSE_{metric}', rmse)
        print(f'RMSE for {metric}: {rmse:.4f}')

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(ground_truth_rescaled)), ground_truth_rescaled, color='green', alpha=0.6, label=f'{metric} (Ground Truth)')
        plt.plot(range(len(predictions_rescaled)), predictions_rescaled, linestyle='--', color='purple', alpha=0.6, label=f'{metric} (Predicted)')
        plt.title(f'{metric} Predictions vs Ground Truth')
        plt.xlabel('Time Steps')
        plt.ylabel(f'{metric} Usage')
        plt.grid()
        plt.legend()
        plot_file = os.path.join('Images', f'{model_type.lower()}_{metric}_predictions_vs_ground_truth.png')
        plt.savefig(plot_file)
        mlflow.log_artifact(plot_file)
        plt.show()

        mlflow.end_run()
