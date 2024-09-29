from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score



def normalization_per_channel(data, seq_length):
    print("norm_channel shape: ", data.shape)
    n_features = data.shape[2]
    data = data.reshape(-1, n_features)

    nor_data = []
    scalers = []
    for i in range(n_features):
        feature = data[:, i].reshape(-1, 1)
        print("Check Feature shape: ", feature.shape)
        scaler = MinMaxScaler(feature_range=(0, 1))
        feature = scaler.fit_transform(feature)
        nor_data.append(feature)
        scalers.append(scaler)
    All_Series = np.hstack(nor_data)
    print("In normalization: ", All_Series.shape)
    return All_Series, scalers



def load_all_csv(root_path, seq_length):

    Bd = []
    count = 0
    print("Extract data Right Now, Please Wait !!!!!!!!!!!")
    for csv_data in os.listdir(root_path):
        #count += 1
        data_path = os.path.join(root_path, csv_data)
        with open(data_path, 'r') as file:
            vm_trace = csv.reader(file, delimiter=';')
            headers = next(vm_trace)
            #print(headers)

            #data_list
            cpu_cores = []
            cpu_capicity = []
            cpu_usage_mhz = []
            cpu_usage_percent = []
            mem_capicity = []
            memory_usage = []
            disk_read = []
            disk_write = []
            net_receive = []
            net_transmit = []

            idx_cpu_cores = headers.index('\tCPU cores')
            idx_cpu_capicity = headers.index('\tCPU capacity provisioned [MHZ]')
            idx_cpu_usage_mhz = headers.index('\tCPU usage [MHZ]')
            idx_cpu_usage_percent = headers.index('\tCPU usage [%]')
            idx_mem_capicity = headers.index('\tMemory capacity provisioned [KB]')
            idx_mem_usage = headers.index('\tMemory usage [KB]')
            idx_disk_read = headers.index('\tDisk read throughput [KB/s]')
            idx_disk_write = headers.index('\tDisk write throughput [KB/s]')
            idx_net_receive = headers.index('\tNetwork received throughput [KB/s]')
            idx_net_transmit = headers.index('\tNetwork transmitted throughput [KB/s]')

            for row in vm_trace:
                #print(float(row[idx_cpu_usage]))
                cpu_cores.append(float(row[idx_cpu_cores]))
                cpu_capicity.append(float(row[idx_cpu_capicity]))
                cpu_usage_mhz.append(float(row[idx_cpu_usage_mhz]))
                cpu_usage_percent.append(float(row[idx_cpu_usage_percent]))
                mem_capicity.append(float(row[idx_mem_capicity]))

                memory_usage.append(float(row[idx_mem_usage]))
                disk_read.append(float(row[idx_disk_read]))
                disk_write.append(float(row[idx_disk_write]))
                net_receive.append(float(row[idx_net_receive]))
                net_transmit.append(float(row[idx_net_transmit]))


            all_data_list = [cpu_usage_percent, memory_usage, disk_write, net_transmit]
            ##Organize data

            feature_numbers = len(all_data_list)
            nd_data = np.array(all_data_list)

            remainder = nd_data.shape[1]%seq_length
            #print(remainder)
            new_nd_data = nd_data[:, :-remainder].reshape(feature_numbers, seq_length, -1)
            new_nd_data = new_nd_data.transpose((2, 1, 0))
            #print(new_nd_data.shape)

            Bd.append(new_nd_data)
        count += new_nd_data.shape[0]

    All_Series = np.vstack(Bd).astype(np.float32)
    print("Check All_Series after vstack: ", All_Series.shape)
    All_Series, scalers = normalization_per_channel(All_Series, seq_length)
    All_Series = All_Series.reshape(-1, seq_length, feature_numbers)
    #Save scalers
    scalers_dict = {}
    scalers_dict["cpu_usage_percent"] = scalers[0]
    scalers_dict["memory_usage"] = scalers[1]
    scalers_dict["disk_write"] = scalers[2]
    scalers_dict["net_transmit"] = scalers[3]

    with open('./data/scalers_dict.pkl', 'wb') as f:
        pickle.dump(scalers_dict, f)

    ##Save Data
    #np.save('./data/workload_series.npy', All_Series)
    print(All_Series.shape)
    return All_Series, scalers_dict




class Workload_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        history_data = self.data[index]
        future_data = self.labels[index]
        
        return history_data, future_data

