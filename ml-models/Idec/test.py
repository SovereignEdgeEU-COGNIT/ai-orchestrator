from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from idec import AE, IDEC
import torch
import numpy as np
from MTS_utils import CreateDataset
import matplotlib.pyplot as plt



Idec_model = IDEC(
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


pth_model = r"./model/Idec_sil_92.pt"
Idec_model.load_state_dict(torch.load(pth_model, map_location=torch.device('cpu')))

print(Idec_model)

Idec_model.eval()



dataset = CreateDataset()
test_data = dataset.testdata
test_label = dataset.test_y
test_data = torch.Tensor(test_data)

with torch.no_grad():
    out = Idec_model(test_data)
    cluster_score = out[1]
    features_z = out[2]

cluster_results = cluster_score.numpy().argmax(1)

feature_matrix = features_z.numpy()
print(feature_matrix.shape)
#print(np.min(cluster_results))
print(cluster_results)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(feature_matrix)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_results)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Clustering')
plt.show()




