import mlflow
import mlflow.pytorch
import os
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
from data import load_data, scale_data, create_sequences

from myLSTM import LSTMModel, train_lstm_model  # Import LSTM
from myFFNN import FFNNModel, train_ffnn_model  # Import FFNN
from myTCN import TCN  # Import TCN
from myGRU import GRUModel, train_gru_model  # Import GRU

import random
import argparse # Import argparse for command line arguments

from rmse import calculate_rmse





# Hyperparameters
model_type = 'FFNN'  # Change this to 'LSTM', 'FFNN', or 'TCN'
input_size = 4  
hidden_size = 64
num_layers = 2  # Only used for LSTM
output_size = 4  # Predicting the next 4 features
sequence_length = 99
prediction_length = 1
batch_size = 64
learning_rate = 0.001
num_epochs = 1
train_test_ratio = 0.8
tcn_channels = [64, 64, 64]  # Channels for TCN layers

# Start MLflow experiment
mlflow.set_experiment("ML Model Comparison")


# Load and preprocess data
print("Data preparation started.....")
filename = 'workload_series.npy'
data_reshaped = load_data(filename)
data_scaled, scaler = scale_data(data_reshaped)
X, y = create_sequences(data_scaled, sequence_length, prediction_length)

train_size = int(len(X) * train_test_ratio)

# Split the data into training and testing sets
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(len(X_test), len(y_test))

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# Create dataset and dataloader
train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# Start an MLflow run
with mlflow.start_run(run_name=f"{model_type} Model Run"):

    # Log hyperparameters
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("input_size", input_size)
    mlflow.log_param("hidden_size", hidden_size)
    mlflow.log_param("num_layers", num_layers if model_type == 'LSTM' else 'N/A')
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)


    # Ensure the 'models' directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Model selection and initialization
    if model_type == 'LSTM':
        # print("Initializing LSTM model...")
        # model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        # criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # # Train the LSTM model
        # print("Training LSTM model...")
        # train_lstm_model(model, train_loader, criterion, optimizer, num_epochs)

        # # Save the LSTM model
        # model_save_path = os.path.join(models_dir, 'lstm_model.pth')
        # torch.save(model.state_dict(), model_save_path)
        # print(f'Model saved to {model_save_path}')

        # Create a new instance of the model
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        
        model_load_path = os.path.join(models_dir, 'lstm_model.pth')
        model.load_state_dict(torch.load(model_load_path))
        print(f'Model loaded from {model_load_path}')

    elif model_type == 'FFNN':
        # print("Initializing FFNN model...")
        # model = FFNNModel(input_size * sequence_length, hidden_size, output_size)  # Input size adjusted for FFNN
        # criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # # Train the FFNN model
        # print("Training FFNN model...")
        # train_ffnn_model(model, train_loader, criterion, optimizer, num_epochs)

        # # Save the FFNN model
        # model_save_path = os.path.join(models_dir, 'ffnn_model.pth')
        # torch.save(model.state_dict(), model_save_path)
        # print(f'Model saved to {model_save_path}')


        # Create a new instance of the FFNN model
        print("Initializing FFNN model...")
        model = FFNNModel(input_size* sequence_length, hidden_size, output_size)

        # Load the FFNN model's state dictionary
        model_load_path = os.path.join(models_dir, 'ffnn_model.pth')
        model.load_state_dict(torch.load(model_load_path))
        print(f'Model loaded from {model_load_path}')
   
    elif model_type == 'GRU':
        # print("Initializing GRU model...")
        # model = GRUModel(input_size, hidden_size, num_layers, output_size)
        # criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # print("Training GRU model...")
        # train_gru_model(model, train_loader, criterion, optimizer, num_epochs)

        # # Save the GRU model
        # model_save_path = os.path.join(models_dir, 'gru_model.pth')
        # torch.save(model.state_dict(), model_save_path)
        # print(f'Model saved to {model_save_path}')


        print("Initializing GRU model...")
        model = GRUModel(input_size, hidden_size, num_layers, output_size)
        
        model_load_path = os.path.join(models_dir, 'gru_model.pth')
        model.load_state_dict(torch.load(model_load_path))
        print(f'Model loaded from {model_load_path}')



    model.eval()
    X_test_tensor = torch.from_numpy(X_test).float()
    random_indices = random.sample(range(len(X_test_tensor)), 100)

    predictions = []
    ground_truth = []
    total_prediction_time = 0

    # Predict for 100 random samples
    for idx in random_indices:
        with torch.no_grad():
            seq = X_test_tensor[idx].unsqueeze(0)
            
            # Measure the prediction time for each sample
            start_time = time.time()

            if model_type == 'FFNN':
                seq = seq.view(seq.size(0), -1)  # Flatten input for FFNN

            pred = model(seq).detach().numpy()

            # End timing and accumulate the time
            end_time = time.time()
            total_prediction_time += (end_time - start_time)

            predictions.append(pred[0])  # Append predicted values
            gt = y_test[idx]
            ground_truth.append(gt)


    # Calculate the average prediction time per instance
    avg_prediction_time = total_prediction_time / len(random_indices)
    print(f'Average Prediction Time per Instance: {avg_prediction_time:.6f} seconds')

    # Log the average prediction time to MLflow
    mlflow.log_metric('avg_prediction_time_per_instance', avg_prediction_time)



    # Rescale the predictions and ground truth back to original scale
    predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, output_size))
    ground_truth_rescaled = scaler.inverse_transform(np.array(ground_truth).reshape(-1, output_size))

    # List of metric names for plotting
    metric_names = ['CPU', 'Memory', 'Disk Write', 'Network Received']

    # Store RMSE values for each feature
    rmse_values = []

    images_dir = 'Images'

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Iterate over all 4 metrics (features) and create separate plots
    for i, metric_name in enumerate(metric_names):
        plt.figure(figsize=(12, 6))
        
        # Plot ground truth for all 100 random instances (feature `i`)
        plt.plot(range(len(ground_truth_rescaled)), ground_truth_rescaled[:, i], color='green', alpha=0.6, 
                 label=f'{metric_name} (Ground Truth)')
        
        # Plot predicted values for all 100 random test instances (feature `i`)
        plt.plot(range(len(predictions_rescaled)), predictions_rescaled[:, i], linestyle='--', color='purple', 
                 alpha=0.6, label=f'{metric_name} (Predicted)')
        
        plt.title(f'{metric_name} Predictions vs Ground Truth for 100 Random Test Samples')
        plt.xlabel('Time Steps')
        plt.ylabel(f'{metric_name} Usage')
        plt.grid()
        plt.legend([f'{metric_name} (Ground Truth)', f'{metric_name} (Predicted)'])
        
        # Save each plot as a separate file
        plot_file = os.path.join(images_dir, f'{model_type.lower()}_{metric_name}_predictions_vs_ground_truth.png')
        plt.savefig(plot_file)
        
        # Log the plot to MLflow
        mlflow.log_artifact(plot_file)

        plt.show()

        # Calculate RMSE for the current feature
        rmse = calculate_rmse(predictions_rescaled[:, i], ground_truth_rescaled[:, i])
        rmse_values.append(rmse)
        print(f'RMSE for {metric_name}: {rmse:.4f}')

        # Log RMSE to MLflow
        mlflow.log_metric(f'RMSE_{metric_name}', rmse)

    # Print overall RMSE values for each feature
    for i, metric_name in enumerate(metric_names):
        print(f'Final RMSE for {metric_name}: {rmse_values[i]:.4f}')
    
    # End the MLflow run
    mlflow.end_run()
