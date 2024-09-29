from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score


def eval_by_silhouette(data, catogories):
    #data = data.cpu().numpy()
    #catogories = catogories.cpu().numpy()
    score = silhouette_score(data, catogories)

    return score


def norm_channel(data):
    print("norm_channel shape: ", data.shape)
    n_features = data.shape[2]

    nor_data = []
    for i in range(n_features):
        feature = data[:, :, i]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        feature = scaler.fit_transform(feature)
        nor_data.append(feature)

    All_Series = np.hstack(nor_data)
    return All_Series



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


            all_data_list = [cpu_cores, cpu_capicity,  cpu_usage_percent, 
                                 mem_capicity, memory_usage,  disk_write,  
                                 net_transmit]
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
        if count > 70000:
            break

    All_Series = np.vstack(Bd).astype(np.float32)
    All_Series = norm_channel(All_Series)

    '''##Normalized in coarse-grained manner
    print(All_Series.shape)
    All_Series = All_Series.astype(np.float32).reshape(-1, int(feature_numbers*seq_length))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    All_Series = scaler.fit_transform(All_Series)
    print(All_Series.dtype)

    print(All_Series.shape)
    print("Max {}, min {}".format(np.max(All_Series), np.min(All_Series)))
    '''

    y = np.ones((All_Series.shape[0],))
    print(All_Series.shape)
    return All_Series, y


class CreateDataset(Dataset):

    def __init__(self):
        data, label = load_all_csv(r"/Users/zhouz/official_repo/ai-orchestrator/src/ml-models/Idec/VMmonitoring", seq_length=112)
        parti_point = int(0.8*data.shape[0])
        self.x = data[:parti_point, :]
        self.y = label[:parti_point]
        self.testdata = data[parti_point:, :]
        self.test_y = label[parti_point:]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx))

if __name__ == "__main__":
    dataset = CreateDataset()
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=False)