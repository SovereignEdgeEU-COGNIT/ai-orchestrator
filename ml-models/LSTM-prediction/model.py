import torch
import torch.nn as nn



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_dim, output_dim)


        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #print("check lstm's out shape: ", out.shape)
        out_1 = self.fc_1(out[:, -1, :])
        #print("check fully-connect's out shape: ", out_1.shape)
        #output = torch.cat([out_1, out_2, out_3, out_4], dim=1)
        #print("check model's out shape: ", output.shape)
        #print("Check out_1 shape: ", out_1.size())
        return out_1

