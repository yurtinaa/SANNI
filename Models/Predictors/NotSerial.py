from torch import nn
import torch
from .BaseModel.Predictor import Predictor


class NotSerialPredictor(nn.Module):
    def __init__(self,
                 forward_predictor: Predictor):
        super().__init__()
        self.forward_predictor = forward_predictor

    def forward(self, x: torch.Tensor):
        x = x.clone()
        input_pred = x.clone()
        index = input_pred != input_pred
        input_pred[index] = 0
        pred = self.forward_predictor(input_pred)
        # print(x[0])

        # print(torch.isnan(x[:, -1]).shape)
        # print(x.shape, input_pred.shape)
        x[:, :-1] = input_pred[:, :-1]
        x[torch.isnan(x)] = pred[torch.isnan(x[:, -1])]
        return x

    def fl(self):
        self.forward_predictor.fl()
