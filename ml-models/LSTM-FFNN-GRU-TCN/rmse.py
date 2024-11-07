from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_rmse(predictions, ground_truth):
    return np.sqrt(mean_squared_error(ground_truth, predictions))