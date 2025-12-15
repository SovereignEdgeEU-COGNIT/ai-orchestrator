import csv
import os
from datetime import datetime

def log_forecast_data(latest_actual, forecast, log_file, current_step, step_interval):
    """
    Log the latest actual and the first-step forecast only.

    Parameters:
    - latest_actual (float): The most recent actual request rate (from Prometheus).
    - forecast (list of float): Forecasted values (should be lam[:1]).
    - log_file (str): Path to CSV file.
    - current_step (int): The current step index (in seconds).
    - step_interval (int): Step duration in seconds (e.g., 5).
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    write_header = not os.path.exists(log_file)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["timestamp", "step_index", "rate_type", "value"])

        # Log only the most recent actual
        writer.writerow([timestamp, current_step, "actual", latest_actual])

        # Log only the forecast for the next step
        if forecast:
            next_step = current_step + step_interval
            writer.writerow([timestamp, next_step, "forecast", forecast[0]])