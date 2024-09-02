# -*- coding: utf-8 -*-

from torch import nn
import torch
from Models.Classifier import Classifier


class Predictor(nn.Module):
    def __init__(self, size_subsequent: int,
                 count_snippet: int,
                 dim: int,
                 classifier: Classifier,
                 snippet_list,
                 device,
                 inside_count=0,
                 batch_norm=False,
                 hidden_dim=32,
                 num_layers=1,
                 kernel=5,
                 with_h=False,
                 backward=False,
                 cell='gru',
                 config=None
                 ):
        super().__init__()
        self.with_h = with_h
        self.backward = backward
        self.size_subsequent = size_subsequent
        self.classifier = classifier
        if classifier is not None:
            self.classifier = self.classifier.eval()

        self.device = device
        self.hidden_dim = hidden_dim
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(dim)
        else:
            self.batch_norm = None
        if snippet_list is not None:
            self.snippet_list = torch.tensor(snippet_list,
                                             device=self.device)
        self.dim = dim

        self.inside_count = inside_count
        self.kernel = kernel
        self.padd = (kernel - 1) // 2

        self.cnns1 = nn.ModuleList([nn.Conv1d(
            in_channels=2,
            out_channels=64,
            padding=self.padd,
            kernel_size=(kernel,)) for i in range(dim)])
        self.cnns2 = nn.ModuleList([nn.Conv1d(
            in_channels=64,
            out_channels=128,
            padding=self.padd,
            kernel_size=(kernel,)) for i in range(dim)])

        self.cnns3 = nn.ModuleList([nn.Conv1d(
            in_channels=128,
            out_channels=256,
            padding=self.padd,
            kernel_size=(kernel,)) for i in range(dim)])

        self.cnns4 = nn.ModuleList([nn.Conv1d(
            in_channels=256,
            out_channels=512,
            padding=self.padd,
            kernel_size=(kernel,)) for i in range(dim)])
        self.last_cnn = 512
        # self.last_cnn = self.cnns4
        self.size_subsequent = self.size_subsequent - 1

        self.gru_dim = nn.ModuleList(
            [nn.GRU(input_size=self.last_cnn,
                    hidden_size=200,
                    bidirectional=False, dropout=0.2,
                    batch_first=True) for i in range(dim)]
        )
        self.last_gru_dim = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(200 * (self.size_subsequent + 1),
                          self.size_subsequent + 1),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            ) for i in range(dim)]
        )
        self.num_layers = num_layers

        rnns_size = 1
        if self.dim > 1:
            rnns_size = self.dim
        self.rnns = nn.GRU(input_size=rnns_size,
                           hidden_size=200,
                           num_layers=self.num_layers,
                           bidirectional=False,
                           dropout=0.2,
                           batch_first=True)

        self.last = nn.Sequential(nn.Linear(200 * (self.size_subsequent + 1), dim),

                                  nn.LeakyReLU())
        self.leakyRelu = nn.LeakyReLU()
        self.init_hidden(1)
        self.first_ar = torch.arange(0,
                                     end=self.dim,
                                     device=self.device)
        self.second_ar = torch.arange(0,
                                      end=self.dim,
                                      device=self.device)

    def forward(self, x):


        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self._augmentation(x)
        x = self._gru_layers(x)
        x = self.__last_layers(x)

        return x

    def _augmentation(self, x):
        # print('inpute', x.shape)
        result_x = torch.zeros(size=(x.shape[0],
                                     self.dim,
                                     x.shape[1]),
                               device=self.device)
        x = x.transpose(2, 1)

        if self.classifier is not None:

            if self.backward:
                snippet = self.classifier(torch.flip(x, [2]))
            else:
                snippet = self.classifier(x)
            snippet = torch.argmax(snippet, dim=1).long()

            snip = self.__snippet_tensor(snippet)
            # last = snip[:, :, -1]

        for i, cnn in enumerate(self.cnns3):
            if self.classifier is not None:
                input_cnn = torch.cat((x[:, i:i + 1, :], snip[:, i:i + 1, :]),
                                      dim=1)
            else:
                input_cnn = x[:, i * 2:i * 2 + 2, :]

            input_cnn = self.leakyRelu(self.cnns1[i](input_cnn))
            input_cnn = self.leakyRelu(self.cnns2[i](input_cnn))
            # if len(self.inside) > 0:
            #     for j, inside in self.inside[i]:
            #         input_cnn = inside(input_cnn)
            # result = self.leakyRelu(cnn(input_cnn))
            input_cnn = self.leakyRelu(cnn(input_cnn))
            result = self.leakyRelu(self.cnns4[i](input_cnn))

            result = result.transpose(1, 2)
            result, _ = self.gru_dim[i](result)
            result = nn.Flatten()(result)
            result = self.last_gru_dim[i](self.leakyRelu(result))
            # result = result.transpose(1, 2)
            # print(result.shape)
            # print(result_x[:, i:i + 1, :].shape)
            result_x[:, i:i + 1, :] = result[:, None, :]
        # print('out', result_x.shape)
        return result_x

    def _gru_layers(self, x):
        # print(x.shape)
        # if self.dim > 1:
        #     x = self.inter_axis_conv(x)
        x = x.transpose(1, 2)
        if self.with_h:
            x, self.h_rnn = self.rnns(x, self.h_rnn)
            self.h_rnn = self.h_rnn
        else:
            x, _ = self.rnns(x)
        x = self.leakyRelu(x)[:, :, :]
        x = nn.Flatten()(x)
        return x

    def __last_layers(self, x):
        x = self.last(x)
        return x

    def __snippet_tensor(self, snippet):
        result = torch.zeros(snippet.shape[0],
                             self.dim,
                             self.snippet_list.shape[2],
                             device=self.device)

        for batch_number in torch.arange(0,
                                         end=snippet.shape[0],
                                         device=self.device):
            for ids in self.second_ar:
                result[batch_number, ids, :] = self.snippet_list[ids,
                                                                 snippet[batch_number, ids]]
        return result

    def __get_snippet(self, number):
        arr = torch.zeros(self.dim,
                          self.size_subsequent,
                          device=self.device)
        for ids, item in enumerate(number):
            arr[ids, :] = self.snippet_list[ids, item]
        return arr

    def init_hidden(self, batch_size):
        self.gru_dim_h = torch.zeros(1, batch_size,
                                     self.hidden_dim).to(self.device).detach()
        self.h_rnn = torch.zeros(1, batch_size,
                                 self.hidden_dim).to(self.device).detach()

    def change_snippet(self, type='change'):
        if self.snippet_list is not None:
            if type == 'change':
                first = self.snippet_list[:, 0, :].clone()
                # print(self.snippet_list.shape)
                # print(self.snippet_list[:, 0, :].mean(), self.snippet_list[:, 1, :].mean())
                self.snippet_list[:, 0, :] = self.snippet_list[:, 1, :].clone()
                self.snippet_list[:, 1, :] = first
                # print(self.snippet_list[:, 0, :].mean(), self.snippet_list[:, 1, :].mean())
            else:
                self.snippet_list[:, :, :] = torch.zeros(self.snippet_list.shape,
                                                         device=self.device)
                # print(self.snippet_list.mean())

    def fl(self):
        for i, rnn in enumerate(self.rnns):
            rnn.flatten_parameters()
