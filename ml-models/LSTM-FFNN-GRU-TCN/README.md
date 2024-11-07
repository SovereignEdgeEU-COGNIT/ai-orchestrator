# Model Testing
To test the models, follow these steps:

## Prerequisites
Ensure you have the required dependencies installed. You can install them using:
    ```bash
    pip install -r requirements.txt
    ```

## Multivariate Model Testing
The multivariate test evaluates models on multiple metrics (CPU, Memory, Disk Write, and Network Received) simultaneously. Follow these steps:
1. Navigate to the directory where the `test.py` script is located.

2. Run the test script using the following command:
    ```bash
    python test.py <model_type>
    ```
    Replace `<model_type>` with one of the following options: `LSTM`, `FFNN`, or `GRU`.

3. The script will load the specified model and evaluate it using the test dataset, generating plots for CPU and Memory predictions.

4. The results, including RMSE values, will be printed to the console, and visualizations will be saved in the `Images` directory.


## Univariate Model Testing
The univariate test evaluates models on a single metric (e.g., CPU, Memory, Disk Write, or Network Received) at a time. Follow these steps:
1. Navigate to the directory where the `test_uni.py` script is located.

2. Run the test script using the following command:
    ```bash
    python test_uni.py <model_type> <metric>
    ```
    Replace `<model_type>` with one of the following options: `LSTM`, `FFNN`, `GRU`, or `TCN`. For now there is only supprot for `FFNN` model.
    Replace `<metric>` with the desired metric for univariate prediction. Options include `CPU`, `Memory`, `Disk Write`, or `Network Received`.

3. The script will load the specified model and evaluate it using the univariate test dataset for the specified metric.

4. The results, will be shown on the screen as images.

# Model Integration
# Single Sequence Integration with FFNNModel_uni

The `FFNNModel_uni` model can be loaded and used to make predictions on a single univariate time series sequence. This guide provides the steps to integrate and make predictions using `FFNNModel_uni`.

## Steps to Use FFNNModel_uni for Single Sequence Prediction

### 1. Load the Model
To begin, load the `FFNNModel_uni` model with the appropriate parameters.

```python
import torch
from myFFNN import FFNNModel_uni  

# Model parameters
sequence_length = 99
hidden_size = 64
output_size = 1

# Initialize and load the model
model = FFNNModel_uni(sequence_length, hidden_size, output_size)
model.load_state_dict(torch.load('models_uni/ffnn_uni_model_<metric>.pth'))
model.eval()  # Set the model to evaluation mode
```

### 2. Prepare the Input Sequence
Ensure the input sequence has the shape `(sequence_length, 1)`, where `sequence_length` is set to 99. The input should be a single sequence to be fed into the model.

```python
import numpy as np

# Example: load or create a single sequence of shape (99, 1)
X_single = np.load('TestSet_uni/X_test_<metric>.npy')[0]  # Select the first test sequence
X_single_tensor = torch.from_numpy(X_single).float().view(sequence_length, 1)  # Shape: (99, 1)

```
### 3. Make a Prediction
Pass the prepared sequence through the model to obtain a prediction.

```python
with torch.no_grad():
    prediction = model(X_single_tensor).item()  # Get single scalar output
print("Prediction:", prediction)
```
### Notes
- Replace `<metric>` with the specific metric name, such as `CPU` or `Memory`.
- This approach provides a straightforward way to load a trained `FFNNModel_uni`, pass a single sequence, and retrieve the prediction output.