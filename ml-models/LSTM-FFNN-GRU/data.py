import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filename):
    """
    Load the .npy file and reshape it into the required format.

    :param filename: str, path to the .npy file
    :return: np.array, reshaped data
    """
    # Load the data
    data = np.load(filename)
    # Reshape the data to (total_samples, 4) where 4 is the number of features
    data_reshaped = data.reshape(-1, 4)
    
    # Optionally, reduce the size for memory constraints
    data_reshaped = data_reshaped[:len(data_reshaped) // 50]  # Use 1/50 of the dataset



    # Convert data to float32 for memory efficiency
    data_reshaped = data_reshaped.astype(np.float32)
    
    return data_reshaped


def rolling_window(a, window):
    b = []
    for i in range(len(a)-window+1):
        b.append([])
        for j in range(window):
            b[-1].append(a[j+i])
    return b
def scale_data(data):
    """
    Scale the data to the range [0, 1].

    :param data: np.array, unscaled data
    :return: np.array, scaled data; scaler object for inverse transformation
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def create_sequences(data, seq_length, pred_length):
    """
    Convert data into sequences for LSTM training.

    :param data: np.array, scaled data
    :param seq_length: int, length of the input sequences
    :param pred_length: int, number of steps to predict
    :return: np.array, np.array: sequences and their corresponding targets
    """
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    return np.array(X), np.array(y)