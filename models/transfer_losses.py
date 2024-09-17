import torch
import torch.nn as nn
from models.loss_funcs import *

class TransferLoss(nn.Module):
    def __init__(self, loss_type='mmd',backbone='unet'):
        super(TransferLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "mmd":
            self.loss_func = MMDLoss()

        elif loss_type == "adv":
            self.loss_func = AdversarialLoss(backbone=backbone)

        elif loss_type == "advmmd":
            self.loss_func = Advmmd(backbone=backbone)
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y: 0 # return 0
    
    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)