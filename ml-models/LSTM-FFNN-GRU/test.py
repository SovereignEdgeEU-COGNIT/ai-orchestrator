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
def load_model(model_type, input_size, hidden_size, output_size, num_layers, sequence_length):
    models_dir = 'models'
    model = None

    if model_type == 'LSTM':
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        model_load_path = os.path.join(models_dir, 'lstm_model.pth')

    elif model_type == 'FFNN':
        model = FFNNModel(input_size * sequence_length, hidden_size, output_size)
        model_load_path = os.path.join(models_dir, 'ffnn_model.pth')

    elif model_type == 'GRU':
        model = GRUModel(input_size, hidden_size, num_layers, output_size)
        model_load_path = os.path.join(models_dir, 'gru_model.pth')

    # Load the model's state dictionary
    if model is not None and os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path))
        print(f'Model loaded from {model_load_path}')
    else:
        raise ValueError(f"Model of type {model_type} is not recognized or model file does not exist.")

    return model

def main(model_type):
    # Hyperparameters
    input_size = 4  
    hidden_size = 64
    num_layers = 2  # Only used for LSTM
    output_size = 4  # Predicting the next 4 features
    sequence_length = 99
    batch_size = 64

    # Load the test set from .npy files
    save_dir = 'TestSet'
    X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'))

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).float()

    # Create dataset and dataloader
    dataset = data.TensorDataset(X_tensor, y_tensor)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = load_model(model_type, input_size, hidden_size, output_size, num_layers, sequence_length)
    model.eval()  # Set the model to evaluation mode

    predictions = []
    ground_truth = []

    # Test the model
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            if model_type == 'FFNN':
                X_batch = X_batch.view(X_batch.size(0), -1)  # Flatten input for FFNN

            preds = model(X_batch).detach().numpy()
            predictions.extend(preds)
            ground_truth.extend(y_batch.numpy())

    # Rescale predictions and ground truth
    scaler = None  # You can load your scaler if necessary
    # predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, output_size))
    # ground_truth_rescaled = scaler.inverse_transform(np.array(ground_truth).reshape(-1, output_size))
    
    # For simplicity, we will assume no scaling is required
    predictions_rescaled = np.array(predictions).reshape(-1, output_size)
    ground_truth_rescaled = np.array(ground_truth).reshape(-1, output_size)

    # Calculate RMSE for each metric
    metric_names = ['CPU', 'Memory', 'Disk Write', 'Network Received']
    rmse_values = []
    for i, metric_name in enumerate(metric_names):
        rmse = calculate_rmse(predictions_rescaled[:, i], ground_truth_rescaled[:, i])
        rmse_values.append(rmse)
        print(f'RMSE for {metric_name}: {rmse:.4f}')

    # Select 100 random indices for visualization
    total_samples = len(predictions_rescaled)
    random_indices = random.sample(range(total_samples), 100)

    # Plot for CPU and Memory using selected indices
    images_dir = 'Images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for i, metric_name in enumerate(metric_names[:2]):  # Only CPU and Memory
        plt.figure(figsize=(12, 6))
        
        # Extract the data for the selected random indices
        ground_truth_selected = ground_truth_rescaled[random_indices, i]
        predictions_selected = predictions_rescaled[random_indices, i]
        
        # Plot ground truth for the current metric
        plt.plot(range(100), ground_truth_selected, color='green', alpha=0.6, 
                 label=f'{metric_name} (Ground Truth)')
        
        # Plot predicted values for the current metric
        plt.plot(range(100), predictions_selected, linestyle='--', color='purple', 
                 alpha=0.6, label=f'{metric_name} (Predicted)')
        
        plt.title(f'{metric_name} Predictions vs Ground Truth for 100 Random Test Samples')
        plt.xlabel('Random Samples')
        plt.ylabel(f'{metric_name} Usage')
        plt.grid()
        plt.legend([f'{metric_name} (Ground Truth)', f'{metric_name} (Predicted)'])
        
        # Save each plot as a separate file
        plot_file = os.path.join(images_dir, f'{model_type.lower()}_{metric_name.lower()}_predictions_vs_ground_truth.png')
        plt.savefig(plot_file)
        
        plt.show()

if __name__ == "__main__":
    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python3 test.py <model_type>")
        sys.exit(1)

    model_type_arg = sys.argv[1]
    valid_model_types = ['LSTM', 'FFNN', 'GRU']

    if model_type_arg not in valid_model_types:
        print(f"Invalid model type. Choose from: {', '.join(valid_model_types)}")
        sys.exit(1)

    main(model_type_arg)
