from torch import nn
import torch
from lib.SANNI_v2.Models.Predictors.BaseModel.Predictor import Predictor


class WithSnippetPredictor(nn.Module):
    def __init__(self,
                 forward_predictor):
        super().__init__()
        self.forward_predictor = forward_predictor

    def forward(self, x: torch.Tensor, reinit=True):
        x = x.clone()
        # print(x[:,-1])

        pred_x = x.clone()
        # print(pred_x[0, -1])
        data = pred_x[:, :, ::2]
        snippet = pred_x[:, :, 1::2]
        # pred_x[:, -1, ::2] = pred_x[:, -1, 1::2]
        data[torch.isnan(data)] = snippet[torch.isnan(data)]
        pred_x[:, :, ::2] = data
        pred_x[:, :, 1::2] = snippet

        pred = self.forward_predictor(pred_x)

        # x = x[:, :, ::2]
        # print(x[:, :, ::2].shape)
        x[:, -1, ::2] = pred
        # print(x[:,-1])

        # pred_x[:, -1, ::2] = x[:, -1]
        return x

    def fl(self):
        self.forward_predictor.fl()
