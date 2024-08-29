from .Predictor import Predictor
from torch import nn
import torch
from lib.SANNI_v2.Models.Classifier import Classifier


class ShortPredictor(Predictor):
    def __init__(self, size_subsequent: int,
                 count_snippet: int,
                 dim: int,
                 classifier: Classifier,
                 snippet_list,
                 device,
                 inside_count=0,
                 batch_norm=False,
                 hidden_dim=128,
                 num_layers=1,
                 kernel=5,
                 with_h=False,
                 backward=False,
                 cell='gru',
                 config=None
                 ):
        super().__init__(size_subsequent,
                         count_snippet,
                         dim,
                         classifier,
                         snippet_list,
                         device,
                         inside_count,
                         batch_norm,
                         hidden_dim,
                         num_layers,
                         kernel,
                         with_h,
                         backward,
                         cell,
                         config
                         )
        self.cnns1 = nn.ModuleList([nn.Sequential(nn.Conv1d(
            in_channels=2,
            out_channels=128,
            padding=self.padd,
            kernel_size=(kernel,))) for i in range(dim)])

        self.cnns2 = nn.ModuleList([nn.Conv1d(
            in_channels=128,
            out_channels=64,
            padding=self.padd,
            kernel_size=(kernel,)) for i in range(dim)])

        self.cnns3 = nn.ModuleList([nn.Sequential(nn.Conv1d(
            in_channels=64,
            out_channels=32,
            padding=self.padd,
            kernel_size=(kernel,)), nn.MaxPool1d(2)) for i in range(dim)])
