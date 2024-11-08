import numpy as np
import torch
import pandas as pd
import time


def inference(model, input):
    predcition = model(input)

    return predcition


if __name__ == "__main__":

    model_path = "./checkpoint/model_lastest.pth"
    device = torch.device('cuda')
    TCN_model = torch.load(model_path, map_location=device)
    TCN_model.eval()
    TCN_model.to(device)

    dummy_input = torch.rand(1000,4,96)
    dummy_input = dummy_input.to(device)

    start_time = time.time()
    prediction = inference(TCN_model, dummy_input)
    end_time = time.time()
    print("latency is: ", end_time - start_time) ### 0.8229

    print("output is: ", prediction.size())
