from torch import nn
import torch
from .Decoder import TimeDecoder
from .Encoder import TimeEncoder


# fixme сделать базовый и вот этот одним
class SAETI(nn.Module):
    def __init__(self, size_seq, n_features, latent_dim, hidden_size, classifier, snippet_list, init=0):
        super().__init__()
        self.classifier = classifier
        self.snippet_list = snippet_list
        self.n_features = n_features
        self.init = init
        # self.seq_len = seq_len
        self.encoder = TimeEncoder(n_features, latent_dim, hidden_size, size_seq)
        self.decoder = TimeDecoder(n_features, latent_dim, hidden_size, size_seq)

    def __snippet_tensor(self, snippet):
        result = torch.zeros(snippet.shape[0],
                             self.n_features,
                             self.snippet_list.shape[2],
                             device=snippet.device)
        # fixme нужен парарлелизм

        for batch_number in torch.arange(0, end=snippet.shape[0], device=snippet.device):
            for ids in torch.arange(0, end=self.n_features, device=snippet.device):
                result[batch_number, ids, :] = self.snippet_list[ids,
                                                                 snippet[batch_number, ids]]
        return result

    def __add_snippet(self, x):
        x = x.transpose(2, 1)
        snippet = self.classifier(x)
        snippet = torch.argmax(snippet, dim=1).long()
        return self.__snippet_tensor(snippet)

    def forward(self, x):

        x = x.clone()
        index = x != x
        x[index] = 0.0
        snip = self.__add_snippet(x)
        lat = self.encoder(x, snip)
        x = self.decoder(lat)
        return x

    def change_snippet(self, type='change'):
        if self.snippet_list is not None:
            if type == 'change':
                first = self.snippet_list[:, 0, :].clone()
                # print(self.snippet_list.shape)
                print(self.snippet_list[:, 0, :].mean(), self.snippet_list[:, 1, :].mean())
                self.snippet_list[:, 0, :] = self.snippet_list[:, 1, :].clone()
                self.snippet_list[:, 1, :] = first
                print(self.snippet_list[:, 0, :].mean(), self.snippet_list[:, 1, :].mean())
            else:
                self.snippet_list[:, :, :] = torch.zeros(self.snippet_list.shape,
                                                         device=self.snippet_list.device)
                # print(self.snippet_list.mean())