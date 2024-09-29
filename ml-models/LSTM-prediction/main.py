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


def data_split(Time_series_data):

    #Time_series_data = np.transpose(Time_series_data, (0,2,1))
    test_set_size = int(np.round(0.2 * Time_series_data.shape[0]))
    train_set_size = Time_series_data.shape[0] - (test_set_size)

    train_data = Time_series_data[:train_set_size, :-4, :]
    train_label = Time_series_data[:train_set_size, -4:, :]

    test_data = Time_series_data[train_set_size:, :-4, :]
    test_label = Time_series_data[train_set_size:, -4:, :]

    print("x_train's shape: ", train_data.shape)
    print("y_train's shape: ", train_label.shape)
    print("x_test's shape: ", test_data.shape)
    print("y_test's shape: ", test_label.shape)

    ### Put in torch dataloader
    train_dataset = Workload_dataset(train_data, train_label)
    test_dataset = Workload_dataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

    return train_loader, test_loader


def data_scaler(data):
    scaler_cpu = MinMaxScaler(feature_range=(0, 1))
    cpu_series = data[:, :, 0]
    cpu_series = scaler_cpu.fit_transform(cpu_series.reshape(-1,1))
    cpu_series = cpu_series.reshape(-1, 100)

    scaler_mem = MinMaxScaler(feature_range=(0, 1))
    mem_series = data[:, :, 1]
    mem_series = scaler_mem.fit_transform(mem_series.reshape(-1,1))
    mem_series = mem_series.reshape(-1, 100)

    scaler_disk = MinMaxScaler(feature_range=(0, 1))
    disk_series = data[:, :, 2]
    disk_series = scaler_disk.fit_transform(disk_series.reshape(-1,1))
    disk_series = disk_series.reshape(-1, 100)
    print("disk_series's shape: ", disk_series.shape)



    All_Series = np.stack((cpu_series, mem_series, disk_series), axis=2)

    print("All_Series's shape: ", All_Series.shape)

    return All_Series, scaler_cpu, scaler_mem, scaler_disk


def build_model(input_dim, hidden_dim, num_layers, output_dim, device):
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, device=device)
    #model = LSTMAttention(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, output_size=output_dim)

    return model

def train(model, num_epochs, train_loader, test_loader, criterion, optimiser, device):

    model = model.to(device)
    
    train_loss_list=[]
    test_loss_list = []
    for i in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0

        for data in train_loader:
            x_train, y_train = data[0].to(device), data[1].to(device)
            y_train_pred = model(x_train)
            y_train = torch.reshape(y_train, (-1, 16))
            loss = criterion(y_train_pred, y_train)
            train_loss_epoch += loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        avg_train_loss = train_loss_epoch / len(train_loader)
        train_loss_list.append(avg_train_loss.item())
        
        ##Evaluation
        model.eval()
        eval_loss_epoch = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                x_test, y_test = test_data[0].to(device), test_data[1].to(device)
                y_test_pred = model(x_test)
                y_test = torch.reshape(y_test, (-1, 16))
                loss = criterion(y_test_pred, y_test)
                eval_loss_epoch += loss
        avg_test_loss = eval_loss_epoch / len(test_loader)
        test_loss_list.append(avg_test_loss.item())
        print("Epoch ", i, "Train Loss: ", round(avg_train_loss.item(), 7), "Test Loss: ", round(avg_test_loss.item(), 7))
            
    ## Save Model
    torch.save(model, "./checkpoint/model_lastest_h128.pth")
    return train_loss_list, test_loss_list

'''
def evaluation(model, test_loader, device):
    model = model.eval()
    test_pred = []
    test_label = []
    with torch.no_grad():
        for test_data in test_loader:
            x_test, y_test = test_data[0].to(device), test_data[1].to(device)
            y_test_pred = model(x_test)

            y_test_pred = y_test_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()

            test_pred.append(y_test_pred)
            test_label.append(y_test)

    #print("check test_pred: ", len(test_pred), test_pred[-1].shape)
    #print("check test_label: ", len(test_label), test_label[-1].shape)
    test_label = np.concatenate(test_label, axis=0)
    test_pred = np.concatenate(test_pred, axis=0)
    print("predcition is: ", test_pred.shape, " | ", test_label.shape)
    rme_value = math.sqrt(mean_squared_error(test_label, test_pred))
    print('RME value: %.2f RMSE' % (rme_value))

    return test_pred, test_label

'''

def evaluation(model, test_loader, device):
    model = model.eval()
    test_pred = []
    test_label = []
    with torch.no_grad():
        for test_data in test_loader:
            x_test, y_test = test_data[0].to(device), test_data[1].to(device)
            y_test_pred = model(x_test)

            y_test_pred = y_test_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()

            test_pred.append(y_test_pred)
            test_label.append(y_test)

    #print("check test_pred: ", len(test_pred), test_pred[-1].shape)
    #print("check test_label: ", len(test_label), test_label[-1].shape)
    test_label = np.concatenate(test_label, axis=0).reshape(-1, 16)
    test_pred = np.concatenate(test_pred, axis=0)
    print("Test label shape: ", test_label.shape, "test_pred shape: ", test_pred.shape)
    print("predcition is: ", test_pred.shape, " | ", test_label.shape)
    #rme_value = math.sqrt(mean_squared_error(test_label, test_pred))
    #print('TOTAL RME value: %.4f RMSE' % (rme_value))

    cpu_label, cpu_pred = test_label[:, :4], test_pred[:, :4]
    cpu_rme_value = math.sqrt(mean_squared_error(cpu_label, cpu_pred))
    print('CPU RME value: %.4f RMSE' % (cpu_rme_value))

    mem_label, mem_pred = test_label[:, 4:8], test_pred[:, 4:8]
    mem_rme_value = math.sqrt(mean_squared_error(mem_label, mem_pred))
    print('MEMORY RME value: %.4f RMSE' % (mem_rme_value))

    disk_label, disk_pred = test_label[:, 8:12], test_pred[:, 8:12]
    disk_rme_value = math.sqrt(mean_squared_error(disk_label, disk_pred))
    print('DISK RME value: %.4f RMSE' % (disk_rme_value))

    net_label, net_pred = test_label[:, 12:], test_pred[:, 12:]
    net_rme_value = math.sqrt(mean_squared_error(net_label, net_pred))
    print('NET RME value: %.4f RMSE' % (net_rme_value))

    print('TOTAL RME value: %.4f RMSE' % ((cpu_rme_value+mem_rme_value+disk_rme_value+net_rme_value)/4))

    return cpu_label, cpu_pred, mem_label, mem_pred, disk_label, disk_pred, net_label, net_pred


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
        All_Series, scalers = load_all_csv(root_data_path, 100) #100 is sequential size
    print("Check All Series shape: ", All_Series.shape)###
    print("Check All scalers: ", scalers.keys())
    #np.random.shuffle(All_Series)


    train_data, test_data = data_split(All_Series)

    input_dim = 4
    hidden_dim = 128 #128 or 256
    num_layers = 2
    predict_steps = 4
    output_dim = input_dim * predict_steps
    num_epochs = 150

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = build_model(input_dim, hidden_dim, num_layers, output_dim, device).to(device)
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.00005)

    train_loss_list, test_loss_list = train(model, num_epochs, train_data, test_data, criterion, optimiser, device)


    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(train_loss_list, color='b', label='train loss curve')
    plt.savefig('./hidden128/CPU_train_loss_curve.jpg')

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(test_loss_list, color='b', label='test loss curve')
    plt.savefig('./hidden128/CPU_test_loss_curve.jpg')

    cpu_label, cpu_pred, mem_label, mem_pred, disk_label, disk_pred, net_label, net_pred = evaluation(model, test_data, device)
    
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

















