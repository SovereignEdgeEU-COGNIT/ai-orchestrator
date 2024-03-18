import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from MTS_utils import load_all_csv
import pickle
from MTS_utils import eval_by_silhouette

class MTS:
    def __init__(self, ts):
        self.ts = ts

    def cov_mat(self, centering=True):
        stdsc = StandardScaler()
        X = self.ts
        X = stdsc.fit_transform(X)
        self.ts = X
        return X.transpose() @ X #@ is matrix multiply, here returns covariance matrix

class CPCA:
    def __init__(self, epsilon=1e-5):
        self.cov = None
        self.epsilon = epsilon
        self.U = None
        self.V = None
        self.S = None

    def fit(self, listMTS):
        if (len(listMTS) > 0):
            P = listMTS[0].cov_mat().shape[1]
            cov_mat = [mat.cov_mat() for mat in listMTS]
            self.cov = sum(cov_mat) / len(cov_mat) #common covariance matrix
            # Add epsilon Id in order to ensure invertibility, invertibility:可逆性
            cov = self.cov + self.epsilon * np.eye(P)
            # Compute SVD
            U, S, V = np.linalg.svd(self.cov)
            # Save SVD
            self.U = U
            self.S = S
            self.V = V

    def pred(self, listMTS, ncp):
        predicted = []
        if (self.U is not None):
            predicted = [elem.ts @ self.U[:, :ncp] for elem in listMTS] #feature tensor of MTS Xi: Pi = Xi*S
            #print("Check feature tensor's shape: ", len(predicted), predicted[0].shape)
        return predicted

    def reconstitution_error(self, listMTS, ncp):
        mse = np.full(len(listMTS), np.inf)
        if (self.U is not None):
            prediction = self.pred(listMTS, ncp)
            reconstit = [elem @ ((self.U)[:, :ncp].transpose()) for elem in prediction]
            mse = [((listMTS[i].ts - reconstit[i]) ** 2).sum() for i in range(len(prediction))]
        return mse, prediction

class Mc2PCA:
    def __init__(self, K, ncp, itermax=1000, conv_crit=1e-5):
        self.K = K
        self.N = None
        self.ncp = ncp
        self.iter_max = itermax
        self.converged = False
        self.CPCA_final = None
        self.conv_crit = conv_crit
        self.pred = None

    def fit(self, X):
        N = len(X)
        print("In fit() : ", N)
        # initialisation
        index_cluster = np.tile(np.arange(self.K), int(N / self.K) + 1)[:N]
        print("index_cluster: ", index_cluster.shape, index_cluster)
        to_continue = True
        i = 0
        old_error = -1

        iteration = 0
        train_sil_score = []
        while to_continue:
            iteration += 1
            # Split all MTS according to the cluster
            # we store it in a list of lists of MTS (each list inside the list corresponding to a cluster)
            MTS_by_cluster = [[X[i] for i in list(np.where(index_cluster == j)[0])] for j in range(self.K)]

            CPCA_by_cluster = [CPCA() for i in range(self.K)]

            # fit by cluster
            [CPCA_by_cluster[i].fit(MTS_by_cluster[i]) for i in range(self.K)]

            res = np.array([cpca.reconstitution_error(X, self.ncp)[0] for cpca in CPCA_by_cluster])
            All_feature_tensors = [cpca.reconstitution_error(X, self.ncp)[1] for cpca in CPCA_by_cluster]
            #print(feature_tensors[0][0].shape)
            # Update index cluster
            index_cluster = res.argmin(axis=0)#choose minimal error
            #print("Check index cluster: ", type(index_cluster), index_cluster)
            
            feature_list = []
            for i in range(N):
                cluster_n = index_cluster[i]
                feature = All_feature_tensors[cluster_n][i]
                feature_list.append(feature)
            feature_array = np.array(feature_list).reshape(N, -1)

            sil_score = eval_by_silhouette(feature_array, index_cluster)
            train_sil_score.append(sil_score)
            print("In Iteration {} silhouette_score is {}".format(iteration, sil_score))

            # new total error
            new_error = res.min(axis=0).sum()
            to_continue = (abs(old_error - new_error) > self.conv_crit) #& (self.iter_max > i)
            self.converged = np.abs(old_error - new_error) < self.conv_crit

            # Updata
            old_error = new_error
            i += 1
        self.CPCA_final = CPCA_by_cluster
        #print("Check Common Space: ", self.CPCA_final[0].U, self.CPCA_final[0].U.shape)

        self.pred = index_cluster
        print(self.pred.shape)

        #save model
        cluster_cax = {}
        for i in range(self.K):
            common_axes = CPCA_by_cluster[i].U[:, :self.ncp]
            cluster_cax[i] = common_axes
        with open('MC2PCA_model.pkl', 'wb') as f:
            pickle.dump(cluster_cax, f)
            print("Model Saved !!!!")

        return index_cluster


    def precision(self, gt_cluster):
        index_cluster = self.pred
        N = gt_cluster.shape[0]
        g = np.unique(gt_cluster)
        nb_g = g.shape[0]

        G = [np.where(gt_cluster == i)[0] for i in range(nb_g)]
        C = [np.where(index_cluster == i)[0] for i in range(self.K)]

        # to handle case where a cluster is empty
        max_part = list()
        for j in range(self.K):
            l = list()
            for i in range(nb_g):
                if len(C[j]) != 0:
                    l.append([np.intersect1d(G[i], C[j]).shape[0] / C[j].shape[0]])
                else:
                    l.append(0)
            max_part.append(np.max(l))
        max_part = np.array(max_part)

        # max_part = np.array([max([np.intersect1d(G[i],C[j]).shape[0]/C[j].shape[0] for i in range(nb_g)]) for j in range(self.K)])
        prop_part = np.array([C[j].shape[0] / N for j in range(self.K)])
        return max_part.dot(prop_part)


def search_ncp(X, K, ncp_list, y_true):
    pres = np.zeros(ncp_list.shape[0])
    for i in range(len(ncp_list)):
        m = Mc2PCA(K, ncp_list[i])
        m.fit(X)
        pres[i] = m.precision(y_true)
    pre = np.max(pres)
    best_ncp = ncp_list[np.argmax(pres)]
    return best_ncp, pre


def train_vmdata():
    cluster = 3
    train_data, test_data = load_all_csv(r"/Users/zhouz/Project/VM_Workload_Predictor/fastStorage/VMmonitoring", seq_length=112)
    print("Train: ", train_data.shape)
    print("Test: ", test_data.shape)

    res = []
    for i in range(train_data.shape[0]):
        sample = train_data[i, :, :]
        sample = MTS(sample)
        res.append(sample)
    
    m = Mc2PCA(K=cluster,ncp=4)
    r = m.fit(res)
    #print("clustering result is: "+str(r))
    
    

if __name__ == '__main__':

    train_vmdata()