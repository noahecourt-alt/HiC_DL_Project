import torch
import torch.nn as nn
import torch.nn.functional as F

class AbsRelativeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Absolute Relative Loss"

    def forward(self, pred, target):
        #formula: |pred - target| / |target|
        numer = torch.sum(torch.abs(pred - target), dim=(1,2,3))
        denominator = torch.sum(torch.abs(target), dim=(1,2,3))
        return torch.mean(numer / denominator)

    def get_name(self):
        return self.name

class MSE_log(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "mse log"
    
    def forward(self, pred, target):
        input_log = torch.sign(pred) * torch.log(torch.abs(pred) + 1)
        target_log = torch.sign(target) * torch.log(torch.abs(target) + 1)
        return F.mse_loss(input_log, target_log)
    
    def get_name(self):
        return self.name


def get_loss(loss: str):
    if loss == 'absrelloss':
        loss_fn = AbsRelativeLoss()
    elif loss == 'mse':
        loss_fn = nn.MSELoss()
    elif loss == 'mselog':
        loss_fn = MSE_log()
    else:
        raise ValueError("Not a valid loss function")
    return loss_fn
