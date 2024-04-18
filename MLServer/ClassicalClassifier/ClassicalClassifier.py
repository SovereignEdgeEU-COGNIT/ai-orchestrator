import pickle
from typing import List
from sklearn.preprocessing import StandardScaler
import numpy as np
from ClassifierInterface import ClassifierInterface
from DBConnector import DBClient, Metric
from FlavorMapper import FlavorMapper
from OnedConnector import OnedConnector

from ClassicalClassifier.MTS_utils import eval_by_silhouette
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class ClassicalClassifier(ClassifierInterface):
    def __init__(self, dbClient: DBClient, onedClient: OnedConnector):
        self.output_size = 3
        self.dbClient = dbClient
        self.flavorMapper = FlavorMapper(self.output_size)
        self.onedClient = onedClient
        self.model = None
        self.seq_length = 112
    
    @staticmethod
    def get_name() -> str:
        return "ClassicalClassifier"
    
    def get_output_size(self) -> int:
        return self.output_size

    def initialize(self):
        model_path = r"./ClassicalClassifier/MC2PCA_model.pkl"

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)


    def predict(self, vm_id: int) -> List[float]:
        
        vm_data = self.dbClient.fetch_latest_metrics(vm_id, 1, self.seq_length)
        
        if len(vm_data) < self.seq_length:
            return self.__predict_using_flavor(vm_id)
        else:
            return self.__predict_using_model(vm_id, vm_data)
        
    
    def preprocess(self, data):
        new_data = []
        for i in range(data.shape[0]):
            sample = data[i, :, :]
            stdsc = StandardScaler()
            sample = stdsc.fit_transform(sample)
            new_data.append(sample)
        
        return new_data
        
    
    def inference(self, data, model, ncp):
        res_error = []
        All_feature_tensors = []
        for key, value in model.items():
            prediction = [mts @ value[:, :ncp] for mts in data]
            reconstit = [elem @ value[:, :ncp].transpose() for elem in prediction]
            mse = [((data[i] - reconstit[i]) ** 2).sum() for i in range(len(prediction))]
            res_error.append(mse)
            All_feature_tensors.append(prediction)
        res = np.array(res_error)
        index_cluster = res.argmin(axis=0)
        #print(index_cluster)
        #print(res)

        # feature_list = []
        # for i in range(len(data)):
        #     cluster_n = index_cluster[i]
        #     feature = All_feature_tensors[cluster_n][i]
        #     feature_list.append(feature)

        # feature_array = np.array(feature_list).reshape(len(data), -1)
        #print(feature_array)

        #sil_score = eval_by_silhouette(feature_array, index_cluster)
        #print("On test data silhouette_score is {}".format(sil_score))
        #pca = PCA(n_components=2)
        #X_pca = pca.fit_transform(feature_array)
        #plt.scatter(X_pca[:, 0], X_pca[:, 1], c=index_cluster)
        #plt.xlabel('Principal Component 1')
        #plt.ylabel('Principal Component 2')
        #plt.title('PCA Clustering')
        #plt.show()

        res = res[:,0]
        # Normalize the result to be between 0 and 1
        #! This normalization takes 1 value to 0 which might underepresent the actual value
        if np.max(res) - np.min(res) == 0:  #! This needs to be adjusted if min == max and min != 0
            res = np.zeros(3)
        else:
            res = (res - np.min(res)) / (np.max(res) - np.min(res))
       
        
        # if res contains nan, replace it with 0
        res = np.nan_to_num(res)
        return res

    def __predict_using_model(self, vm_id: int, vm_data: List[Metric]) -> List[float]:
        vm = self.dbClient.fetch_vm(vm_id)
        
        if len(vm) != 1 or self.model is None:
            return self.__predict_using_flavor(vm_id)
        
        vm = vm[0]
        
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

        data_preprocessed = self.preprocess(data)
        # Convert the 2D list to a tensor
        results = self.inference(data_preprocessed, self.model, ncp=1)
        return results.tolist()
        
    def __predict_using_flavor(self, vm_id: int) -> List[float]:
        flavor = self.onedClient.get_vm_flavor(vm_id)
        
        return self.flavorMapper.get_flavor(flavor)