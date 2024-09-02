from torch import nn
import copy
import torch


class BaseMultivarite(nn.Module):
    def __init__(self,
                 BasePredictor: nn.Module,
                 dim=1):
        super().__init__()
        self.dim = dim
        self.models = BasePredictor

    def forward(self, x):
       # print(x.shape)
        for i in torch.arange(self.dim):
          #  print(x[:, :, i, None].shape)
            x[:, -1, i] = self.models[i](x[:, :-1, i, None])[:, -1]
        return x
