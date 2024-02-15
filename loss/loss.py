import torch

def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

def MAPE(y_true, y_pred):
    epsilon = 1e-8
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true+epsilon))) * 100

    return mape