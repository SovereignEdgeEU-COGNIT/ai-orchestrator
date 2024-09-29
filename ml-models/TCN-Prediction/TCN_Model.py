import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        clip module
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        One TCN block

        :param n_inputs: int, input channel numbers
        :param n_outputs: int, output channel numbers
        :param kernel_size: int, conv kernel size
        :param stride: int, 
        :param dilation: int, dilation coefficient
        :param padding: int, padding coefficient
        :param dropout: float, dropout ratio
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, output_size, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs: int， input channels
        :param channels: list，hidden channels of each layer
        :param kernel_size: int, size of convolution channels
        :param dropout: float, drop_out ratio
        """
        super(TCN, self).__init__()
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # dilation coefficient：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # input channels at each layer
            out_channels = num_channels[i]  # the output channels of each layer
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        print("Hoe many tCN blocks: ", len(layers))#3

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x = self.network(x)
        out = self.fc(x[:,:,-1])
        return out



if __name__ == "__main__":

     model = TCN(num_inputs=4, output_size=4,num_channels=[32, 32, 32])
     model.train()
     #print("Check Model: ", model)
     
     input = np.random.randn(32, 4, 100).astype(np.float32)
     input_tensor = torch.from_numpy(input)
     #print("Input: ", input)
     output = model(input_tensor)
     print(output.shape)
