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