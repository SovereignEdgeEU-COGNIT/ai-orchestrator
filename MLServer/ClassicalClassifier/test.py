import pickle
from sklearn.preprocessing import StandardScaler
from MTS_utils import load_all_csv
import numpy as np
from MTS_utils import eval_by_silhouette
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def preprocess(data):
    new_data = []
    for i in range(data.shape[0]):
        sample = data[i, :, :]
        stdsc = StandardScaler()
        sample = stdsc.fit_transform(sample)
        new_data.append(sample)
    
    return new_data

def inference(data, model, ncp):
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

    feature_list = []
    for i in range(len(data)):
        cluster_n = index_cluster[i]
        feature = All_feature_tensors[cluster_n][i]
        feature_list.append(feature)

    feature_array = np.array(feature_list).reshape(len(data), -1)

    sil_score = eval_by_silhouette(feature_array, index_cluster)
    print("On test data silhouette_score is {}".format(sil_score))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(feature_array)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=index_cluster)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Clustering')
    plt.show()

    return index_cluster


model_path = r"./ClassicalClassifier/MC2PCA_model.pkl"

with open(model_path, 'rb') as file:
    model = pickle.load(file)

test_samples = preprocess(test_data)
results = inference(test_samples, model, ncp=1)
print("Clustering Results: ", results)