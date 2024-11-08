import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from TCN_Model import TCN
from dataset import load_all_csv, Workload_dataset
import time, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from sklearn.metrics import mean_squared_error
import pickle
from main import data_split, data_scaler, evaluation


#indexs:  [8967, 10829, 5735, 16448, 11357, 15307, 10854, 282, 14062, 11271] #random indexes for comparison

if __name__ == '__main__':


    #root_data_path = r"/Users/zhouz/Project/VMmonitoring"
    if os.path.exists("./data/workload_series.npy") and os.path.exists("./data/scalers_dict.pkl"):
        print("Load Normalized series data from ./data/workload_series.npy")
        print("Load saved scaler from ./data/scalers_dict.pkl")
        All_Series = np.load('/proj/zhou-cognit/users/x_zhozh/project/workload-prediction/data/workload_series.npy')
        with open('./data/scalers_dict.pkl', 'rb') as f:
            scalers = pickle.load(f)
    else:
        print("Load data from original *.CSV file")
        root_data_path = r"/proj/zhou-cognit/users/x_zhozh/project/faststorage/VMmonitoring"
        All_Series, scalers = load_all_csv(root_data_path, 100)
        np.random.shuffle(All_Series)
        np.save('./data/workload_series.npy', All_Series)
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
    model_path = "./checkpoint/model_lastest.pth"
    TCN_model = torch.load(model_path)
    TCN_model.eval()
    TCN_model.to(device)
    

    train_data, test_data = data_split(All_Series)

    cpu_label, cpu_pred, mem_label, mem_pred, disk_label, disk_pred, net_label, net_pred = evaluation(TCN_model, test_data, device)

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
    #indexs = [8967, 10829, 5735, 16448, 11357, 15307, 10854, 282, 14062, 11271]
    print("indexs: ", indexs)

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
    
    '''
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(cpu_pred_v, color='b')
    axs[0, 0].plot(cpu_label_v, color='g')
    axs[0, 0].set_title('cpu_usage_percent')

    axs[0, 1].plot(mem_label_v, color='b')
    axs[0, 1].plot(mem_pred_v, color='g')
    axs[0, 1].set_title('memory_usage')

    axs[1, 0].plot(disk_label_v, color='b')
    axs[1, 0].plot(disk_pred_v, color='g')
    axs[1, 0].set_title('disk_write')

    axs[1, 1].plot(net_label_v, color='b')
    axs[1, 1].plot(net_pred_v, color='g')
    axs[1, 1].set_title('net_transmit')

    plt.tight_layout()
    plt.savefig('comparision_evaluation.jpg', dpi=300)
    '''
