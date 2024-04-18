
from typing import List
from ClassifierInterface import ClassifierInterface
from sklearn.decomposition import PCA
from DBConnector import DBClient, Metric

import torch
import numpy as np
from DLClassifier.idec import IDEC
from sklearn.preprocessing import MinMaxScaler

from FlavorMapper import FlavorMapper
from OnedConnector import OnedConnector

from enum import Enum

class DLClassifierType(Enum):
    IR = "intermediateRepresentation"
    CLASSIFIER = "classifier"

class DLClassifier(ClassifierInterface):
    def __init__(self, dbClient: DBClient, onedClient: OnedConnector, type: DLClassifierType):
        self.dbClient = dbClient
        self.Idec_model = None
        self.seq_length = 112
        self.onedClient = onedClient
        self.type = type
        self.output_size = 10 if type == DLClassifierType.IR else 3

        self.flavorMapper = FlavorMapper(self.output_size)
        
    @staticmethod
    def get_name(type: DLClassifierType) -> str:
        return "DLClassifier"if type == DLClassifierType.CLASSIFIER else "DLIR"
    
    def get_output_size(self) -> int:
        return self.output_size

    def initialize(self):
                
        self.Idec_model = IDEC(
                n_enc_1=500,
                n_enc_2=500,
                n_enc_3=1000,
                n_dec_1=1000,
                n_dec_2=500,
                n_dec_3=500,
                n_input=784,
                n_z=10,
                n_clusters=3,
                alpha=1.0,
                pretrain_path='')


        pth_model = r"./DLClassifier/model/Idec_sil_92.pt"
        self.Idec_model.load_state_dict(torch.load(pth_model, map_location=torch.device('cpu')))
        print(self.Idec_model)

        self.Idec_model.eval()
        print("DLClassifier loaded")

    def predict(self, vm_id: int) -> List[float]:
        
        vm_data = self.dbClient.fetch_latest_metrics(vm_id, 1, self.seq_length)
        
        if len(vm_data) < self.seq_length:
            return self.__predict_using_flavor(vm_id)
        else:
            return self.__predict_using_idec(vm_id, vm_data)
        
    
    #! The MinMaxScaler doesn't scale the data as expected, it will only scale the VM metrics relative to itself, not the entire dataset
    def __norm_channel(self, data):
        #print("norm_channel shape: ", data.shape)
        n_features = data.shape[2]

        nor_data = []
        for i in range(n_features):
            feature = data[:, :, i]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            feature = scaler.fit_transform(feature)
            nor_data.append(feature)

        All_Series = np.hstack(nor_data)
        return All_Series

    def __predict_using_idec(self, vm_id: int, vm_data: List[Metric]) -> List[float]:
        vm = self.dbClient.fetch_vm(vm_id)
        
        if len(vm) != 1 or self.Idec_model is None:
            return self.__predict_using_flavor(vm_id)
        
        vm = vm[0]
        
        #! Capacity can't be fetched like this, should get from prom instead
        cpu_capacity_mhz = 3200 * vm.total_cpu
        mem_capacity = vm.total_mem

        # Initialize an empty list to store each metric row
        all_data_matrix = []

        for metric in vm_data:
            row = [
                vm.total_cpu,  # cpu_cores, assuming this is constant across metrics
                cpu_capacity_mhz,  # cpu_capacity, also constant
                metric.cpu,  # Assuming a constant usage percent; might need to adjust
                mem_capacity,  # mem_capacity, constant
                metric.memory,  # memory_usage_percent, constant
                metric.disk_write,  # disk_write from current metric
                metric.net_tx  # net_transmit from current metric
            ]
            all_data_matrix.append(row)
        data = np.array(all_data_matrix)
        
        # reshape the data to 3D
        data = data.reshape(1, self.seq_length, 7)
        #print(data)

        data_normalized_vstacked = self.__norm_channel(data)
        # Convert the 2D list to a tensor
        data_tensor = torch.tensor(data_normalized_vstacked, dtype=torch.float32)
        
        with torch.no_grad():
            out = self.Idec_model(data_tensor)
            cluster_score = out[1]
            features_z = out[2]
            
        #print("Cluster Score: ", cluster_score)
        
        #! These normalization takes 1 value to 0 which might underepresent the actual value
        if self.type == DLClassifierType.IR:
            features_z = (features_z - features_z.min()) / (features_z.max() - features_z.min())
            output = features_z.cpu().detach().numpy().tolist()[0]
        else:
            cluster_score = (cluster_score - cluster_score.min()) / (cluster_score.max() - cluster_score.min())
            output = cluster_score.cpu().detach().numpy().tolist()[0]
        # Normalize feature_z between 0 and 1
        
        flavor = self.onedClient.get_vm_flavor(vm_id)
        self.flavorMapper.update_flavor(flavor, output)
        
        return output
    
    def __predict_using_flavor(self, vm_id: int) -> List[float]:
        flavor = self.onedClient.get_vm_flavor(vm_id)
        
        return self.flavorMapper.get_flavor(flavor)
        
