import numpy as np
import torch
import pandas as pd
from model import LSTM
import time

def inference(model, input):
    predcition = model(input)

    return predcition


if __name__ == "__main__":

    model_path = "./checkpoint/model_lastest_h128.pth"
    device = torch.device('cuda')
    LSTM_model = torch.load(model_path,map_location=device)
    LSTM_model.eval()
    LSTM_model.to(device)

    dummy_input = torch.rand(1000,96,4)
    dummy_input = dummy_input.to(device)

    start_time = time.time()
    prediction = inference(LSTM_model, dummy_input)
    end_time = time.time()
    print("latency is: ", end_time - start_time) ### 0.06s

    print("output is: ", prediction.size())