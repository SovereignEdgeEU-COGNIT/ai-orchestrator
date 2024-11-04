import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from myLSTM import LSTMModel
from myFFNN import FFNNModel_uni  # Assume FFNNModel_uni is for univariate
from myGRU import GRUModel
from rmse import calculate_rmse

# Function to load the model based on model type
def load_model(model_type, sequence_length, hidden_size, output_size, num_layers, metric):
    models_dir = 'models_uni'
    model = None
    model_lower = model_type.lower()
    metric_lower = metric.lower()

    # Include metric name in the filename to load the correct model
    model_filename = f"{model_lower}_model_{metric_lower}.pth"
    model_load_path = os.path.join(models_dir, model_filename)

    # Initialize the model based on the model type
    if model_type == 'LSTM':
        model = LSTMModel(1, hidden_size, num_layers, output_size)
    elif model_type == 'FFNN_uni':
        model = FFNNModel_uni(sequence_length, hidden_size, output_size)
    elif model_type == 'GRU':
        model = GRUModel(1, hidden_size, num_layers, output_size)
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")

    # Load the model's state dictionary if the file exists
    if model is not None and os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path))
        model.eval()  # Set model to evaluation mode
        print(f'Model loaded from {model_load_path}')
    else:
        raise ValueError(f"Model file '{model_filename}' does not exist in '{models_dir}'.")

    return model

def main(model_type, metric):
    # Hyperparameters
    sequence_length = 99
    hidden_size = 64
    num_layers = 2  # Only used for LSTM and GRU
    output_size = 1  # Predicting a single feature

    # Load the test set from .npy files
    save_dir = 'TestSet_uni'
    metric_lower = metric.lower()
    X_test = np.load(os.path.join(save_dir, f'X_test_{metric_lower}.npy'))  # Univariate test data
    y_test = np.load(os.path.join(save_dir, f'y_test_{metric_lower}.npy'))

    # Select a single sequence for testing
    X_single = X_test[0]  # First sequence in the test set
    y_single = y_test[0]  # Corresponding target value

    # Convert to PyTorch tensor
    X_single_tensor = torch.from_numpy(X_single).float().view(sequence_length, 1)  # Reshape to (99, 1)

    # Load the model
    model = load_model(model_type, sequence_length, hidden_size, output_size, num_layers, metric)

    # Make prediction
    with torch.no_grad():
        prediction = model(X_single_tensor).item()  # Get single scalar output

    # Ensure predictions and ground truth are 1D for RMSE calculation
    prediction = np.array([prediction]).reshape(-1)
    y_single = np.array([y_single]).reshape(-1)

    # Calculate RMSE for this single prediction (for illustrative purposes)
    rmse = calculate_rmse(prediction, y_single)
    print(f'RMSE for {metric}: {rmse:.4f}')
    print(f'Prediction: {prediction[0]:.4f}, Ground Truth: {y_single[0]:.4f}')

    # Plot the result
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [y_single[0], y_single[0]], color='green', linestyle='-', label='Ground Truth')
    plt.plot([0, 1], [y_single[0], prediction[0]], color='purple', linestyle='--', label='Prediction')
    plt.title(f'{metric} Prediction vs Ground Truth')
    plt.ylabel(f'{metric} Usage')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test_uni_single.py <model_type> <metric>")
        sys.exit(1)

    model_type_arg = sys.argv[1]
    metric_arg = sys.argv[2]
    valid_model_types = ['LSTM', 'FFNN_uni', 'GRU']
    valid_metrics = ['CPU', 'Memory', 'Disk Write', 'Network Received']

    if model_type_arg not in valid_model_types:
        print(f"Invalid model type. Choose from: {', '.join(valid_model_types)}")
        sys.exit(1)
    
    if metric_arg not in valid_metrics:
        print(f"Invalid metric. Choose from: {', '.join(valid_metrics)}")
        sys.exit(1)

    main(model_type_arg, metric_arg)
