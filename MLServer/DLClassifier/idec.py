from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import matplotlib.pyplot as plt
from DLClassifier.MTS_utils import CreateDataset, eval_by_silhouette
#from model_package import AE, IDEC


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path='./cognit_model/encoder.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))#(3, 10)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', path)

    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q, z



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    ae_loss_list = []
    for epoch in range(200):#200 orig
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        avg_loss = total_loss / (batch_idx + 1)
        ae_loss_list.append(avg_loss)

        print("epoch {} loss={:.4f}".format(epoch,avg_loss))
        torch.save(model.state_dict(), args.pretrain_path)

    print("model saved to {}.".format(args.pretrain_path))
    ##loss curve
    plt.figure()
    plt.plot(ae_loss_list, color='b', label='ae loss curve')
    plt.savefig('ae_loss_curve.png')



def train_idec():

    model = IDEC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)

    #model.pretrain('data/encoder.pkl')
    model.pretrain(path = '')

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = dataset.x
    print("{} train samples".format(data.shape[0]))
    y = dataset.y
    data = torch.Tensor(data).to(device)
    test_data = dataset.testdata
    print("{} test samples".format(test_data.shape[0]))
    test_data = torch.Tensor(test_data).to(device)

    x_bar, hidden = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    si_score = eval_by_silhouette(hidden.data.cpu().numpy(), y_pred)
    print("sil score={:.4f}".format(si_score))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    idec_loss_list = []
    sil_score_list = []
    test_sil_score_list = []
    for epoch in range(100):

        
        if epoch % args.update_interval == 0:

            _, tmp_q, z_ = model(data)# _:x_bar, tmp_q:q
            #print("check _, tmp_q: ", _.shape, " ", tmp_q.shape)

            # update target distribution p
            tmp_q = tmp_q.data
            #print("check tmp_q: ", tmp_q)
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            #print("check y_pred: ", y_pred)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            sil_score = eval_by_silhouette(z_.cpu().detach().numpy(), y_pred_last)
            sil_score_list.append(sil_score)

            res = model(test_data)
            test_z = res[2]
            test_q = res[1]
            test_q_pred = test_q.cpu().detach().numpy().argmax(1)
            test_sil_score = eval_by_silhouette(test_z.cpu().detach().numpy(), test_q_pred)
            test_sil_score_list.append(test_sil_score)

            print("train_silcore is : ", sil_score, "test sil_score is :", test_sil_score)
            #print("y is: ", y) # y is [5 0 4 ... 4 5 6]
  
            print("Training IDEC {}".format(epoch))

        all_loss = 0.
        for batch_idx, (x, _, idx) in enumerate(train_loader):

            x = x.to(device)
            idx = idx.to(device)

            x_bar, q, z = model(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = args.gamma * kl_loss + reconstr_loss
            all_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch {} loss={:.4f}".format(epoch,
                                            all_loss / (batch_idx + 1)))
        idec_loss_list.append(all_loss / (batch_idx + 1))
    

    torch.save(model.state_dict(), "./Idec.pt")
    #torch.save(model, "./Idec.pth")
    #idec loss curve
    plt.figure()
    plt.plot(idec_loss_list, color='b', label='idec loss curve')
    plt.savefig('idec_loss_curve.png')
    #sil score curve-->train
    plt.figure()
    plt.plot(sil_score_list, color='b', label='train sil score curve')
    plt.savefig('sil_score_curve.png')
    #sil score curve-->test
    plt.figure()
    plt.plot(test_sil_score_list, color='b', label='test sil score curve')
    plt.savefig('test_sil_score_list.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='model/encoder.pkl')
    parser.add_argument('--gamma', default=0.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)

    args = parser.parse_args()
    args.mps = torch.backends.mps.is_available()
    print("use mps: {}".format(args.mps))
    device = torch.device("mps" if args.mps else "cpu")

    args.pretrain_path = 'model/encoder.pkl'
    args.n_clusters = 3
    args.n_input = 784
    dataset = CreateDataset()

    print(args)
    train_idec()
    print("Training finished")