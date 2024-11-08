import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import LSTM, LSTMAttention
from dataset import load_all_csv, Workload_dataset
import time, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from sklearn.metrics import mean_squared_error
import pickle
from main import data_split, data_scaler, evaluation


if __name__ == '__main__':


    #root_data_path = r"/Users/zhouz/Project/VMmonitoring"
    if os.path.exists("./data/workload_series.npy") and os.path.exists("./data/scalers_dict.pkl"):
        print("Load Normalized series data from ./data/workload_series.npy")
        print("Load saved scaler from ./data/scalers_dict.pkl")
        All_Series = np.load('./data/workload_series.npy')
        with open('./data/scalers_dict.pkl', 'rb') as f:
            scalers = pickle.load(f)
    else:
        print("Load data from original *.CSV file")
        root_data_path = r"/proj/zhou-cognit/users/x_zhozh/project/faststorage/VMmonitoring"
        All_Series, scalers = load_all_csv(root_data_path, 100)
    print("Check All Series shape: ", All_Series.shape)###
    print("Check All scalers: ", scalers.keys())

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    ##Load model
    model_path = "./checkpoint/model_lastest_h128.pth"
    LSTM_model = torch.load(model_path)
    LSTM_model.eval()
    LSTM_model.to(device)
    

    train_data, test_data = data_split(All_Series)

    cpu_label, cpu_pred, mem_label, mem_pred, disk_label, disk_pred, net_label, net_pred = evaluation(LSTM_model, test_data, device)
    
    cpu_label = np.clip(cpu_label, 0, None)
    cpu_pred = np.clip(cpu_pred, 0, None)
    mem_label = np.clip(mem_label, 0, None)
    mem_pred = np.clip(mem_pred, 0, None)
    disk_label = np.clip(disk_label, 0, None)
    disk_pred = np.clip(disk_pred, 0, None)
    net_label = np.clip(net_label, 0, None)
    net_pred = np.clip(net_pred, 0, None)

    ##Get random index for easier visualization
    indexs = np.random.randint(1, cpu_label.shape[0], size=10).tolist()
    print("indexs: ", indexs)

    cpu_pred_v = cpu_pred[indexs].flatten() # shape = (10,4) => (40,)
    cpu_label_v = cpu_label[indexs].flatten()
    mem_label_v = mem_label[indexs].flatten() # shape = (10,4) => (40,)
    mem_pred_v = mem_pred[indexs].flatten()
    disk_label_v = disk_label[indexs].flatten() # shape = (10,4) => (40,)
    disk_pred_v = disk_pred[indexs].flatten()
    net_label_v = net_label[indexs].flatten() # shape = (10,4) => (40,)
    net_pred_v = net_pred[indexs].flatten()



    ##Visualization
    cpu_pred_v = cpu_pred[indexs].flatten() # shape = (10,4) => (40,)
    #print("Check cpu_pred_v: ", cpu_pred_v)
    cpu_label_v = cpu_label[indexs].flatten()
    mem_label_v = mem_label[indexs].flatten() # shape = (10,4) => (40,)
    mem_pred_v = mem_pred[indexs].flatten()
    disk_label_v = disk_label[indexs].flatten() # shape = (10,4) => (40,)
    disk_pred_v = disk_pred[indexs].flatten()
    net_label_v = net_label[indexs].flatten() # shape = (10,4) => (40,)
    net_pred_v = net_pred[indexs].flatten()

    
    plt.plot(cpu_pred_v,  color='blue', linewidth=2)
    plt.plot(cpu_label_v, color='green', linewidth=2)  
    plt.title('Cpu Usage')
    plt.xlabel('Time steps')
    plt.ylabel('Cpu Usage')
    plt.savefig('./visualization_result/cpu_comparision.jpg', dpi=300)
    plt.close()

    plt.plot(mem_label_v,  color='blue', linewidth=2)
    plt.plot(mem_pred_v, color='green', linewidth=2)  
    plt.title('Memory Usage')
    plt.xlabel('Time steps')
    plt.ylabel('Memory Usage')
    plt.savefig('./visualization_result/Mem_comparision.jpg', dpi=300)
    plt.close()

    plt.plot(disk_label_v,  color='blue', linewidth=2)
    plt.plot(disk_pred_v, color='green', linewidth=2)  
    plt.title('Disk Write')
    plt.xlabel('Time steps')
    plt.ylabel('Disk Write')
    plt.savefig('./visualization_result/Disk_comparision.jpg', dpi=300)
    plt.close()

    plt.plot(net_label_v,  color='blue', linewidth=2)
    plt.plot(net_pred_v, color='green', linewidth=2)  
    plt.title('Network received')
    plt.xlabel('Time steps')
    plt.ylabel('Network received')
    plt.savefig('./visualization_result/Net_comparision.jpg', dpi=300)
    plt.close()
    
    print("Saved at visualization result")