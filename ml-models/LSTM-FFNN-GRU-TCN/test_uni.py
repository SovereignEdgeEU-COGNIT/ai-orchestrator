import os
import sys
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from myLSTM import LSTMModel
from myFFNN import FFNNModel
from myGRU import GRUModel
from rmse import calculate_rmse
import random

# Function to load the model based on model type
def load_model(model_type, input_size, hidden_size, output_size, num_layers, sequence_length, metric):
    models_dir = 'models_uni'
    model = None
    model_lower = model_type.lower()  # Convert model type to lowercase
    metric_lower = metric.lower()     # Convert metric to lowercase

    model_filename = f"{model_lower}_uni_model_{metric_lower}.pth"
    model_load_path = os.path.join(models_dir, model_filename)

    # Initialize the model based on the model type
    if model_type == 'LSTM':
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    elif model_type == 'FFNN':
        model = FFNNModel(sequence_length, hidden_size, output_size)  # Set input size to `sequence_length`
    elif model_type == 'GRU':
        model = GRUModel(input_size, hidden_size, num_layers, output_size)
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")

    # Load the model's state dictionary if the file exists
    if model is not None and os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path))
        print(f'Model loaded from {model_load_path}')
    else:
        raise ValueError(f"Model file '{model_filename}' does not exist in '{models_dir}'.")

    return model

def main(model_type, metric):
    # Hyperparameters
    input_size = 1  # For univariate time series
    hidden_size = 64
    num_layers = 2  # Only used for LSTM and GRU
    output_size = 1  # Predicting a single feature
    sequence_length = 99
    batch_size = 64

    # Load the test set from .npy files
    save_dir = 'TestSet_uni'
    metric_lower = metric.lower()  # Convert metric name to lowercase
    X_test = np.load(os.path.join(save_dir, f'X_test_{metric_lower}.npy'))  # Should be univariate
    y_test = np.load(os.path.join(save_dir, f'y_test_{metric_lower}.npy'))

    # Print shape of X_test for debugging
    print("Shape of X_test:", X_test.shape)

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).float()

    # Create dataset and dataloader
    dataset = data.TensorDataset(X_tensor, y_tensor)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = load_model(model_type, input_size, hidden_size, output_size, num_layers, sequence_length, metric)
    model.eval()  # Set the model to evaluation mode

    predictions = []
    ground_truth = []

    # Test the model
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            print("Shape of X_batch before reshaping:", X_batch.shape)  # Debugging statement
            
            if model_type == 'FFNN':
                # Flatten input for FFNN to match (batch_size, sequence_length)
                X_batch = X_batch.view(X_batch.size(0), sequence_length)  # Flatten for FFNN input
            
            print("Shape of X_batch after reshaping:", X_batch.shape)  # Debugging statement

            preds = model(X_batch).detach().numpy()
            predictions.extend(preds)
            ground_truth.extend(y_batch.numpy())


    # Rescale predictions and ground truth (if needed)
    scaler = None  # Use a scaler if necessary
    predictions_rescaled = np.array(predictions).reshape(-1, output_size)
    ground_truth_rescaled = np.array(ground_truth).reshape(-1, output_size)

    # Calculate RMSE for the specified metric
    rmse = calculate_rmse(predictions_rescaled, ground_truth_rescaled)
    print(f'RMSE for {metric}: {rmse:.4f}')

    # Plot the results for the specified metric
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth and predicted values for visualization
    plt.plot(range(len(ground_truth_rescaled)), ground_truth_rescaled, color='green', alpha=0.6, 
             label=f'{metric} (Ground Truth)')
    plt.plot(range(len(predictions_rescaled)), predictions_rescaled, linestyle='--', color='purple', 
             alpha=0.6, label=f'{metric} (Predicted)')
    
    plt.title(f'{metric} Predictions vs Ground Truth')
    plt.xlabel('Samples')
    plt.ylabel(f'{metric} Usage')
    plt.grid()
    plt.legend([f'{metric} (Ground Truth)', f'{metric} (Predicted)'])
    
    # Save the plot
    images_dir = 'Images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    plot_file = os.path.join(images_dir, f'{model_type.lower()}_{metric_lower}_predictions_vs_ground_truth.png')
    plt.savefig(plot_file)
    plt.show()

if __name__ == "__main__":
    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 test_uni.py <model_type> <metric>")
        sys.exit(1)

    model_type_arg = sys.argv[1]
    metric_arg = sys.argv[2]
    valid_model_types = ['LSTM', 'FFNN', 'GRU']
    valid_metrics = ['CPU', 'Memory', 'Disk Write', 'Network Received']

    if model_type_arg not in valid_model_types:
        print(f"Invalid model type. Choose from: {', '.join(valid_model_types)}")
        sys.exit(1)
    
    if metric_arg not in valid_metrics:
        print(f"Invalid metric. Choose from: {', '.join(valid_metrics)}")
        sys.exit(1)

    main(model_type_arg, metric_arg)
