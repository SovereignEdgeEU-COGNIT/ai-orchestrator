#### Model Testing
To test the models, follow these steps:

1. Ensure you have the required dependencies installed. You can install them using:
    ```bash
    pip install -r requirements.txt
    ```

2. Navigate to the directory where the `test.py` script is located.

3. Run the test script using the following command:
    ```bash
    python test.py <model_type>
    ```
    Replace `<model_type>` with one of the following options: `LSTM`, `FFNN`, or `GRU`.

4. The script will load the specified model and evaluate it using the test dataset, generating plots for CPU and Memory predictions.

5. The results, including RMSE values, will be printed to the console, and visualizations will be saved in the `Images` directory.