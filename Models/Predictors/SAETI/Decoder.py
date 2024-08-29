from torch import nn
import torch


class TimeDecoder(nn.Module):
    def __init__(self, n_features, latent_dim, hidden_size, size_seq,

                 kernel=5):
        super().__init__()

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.size_seq = size_seq
        self.padd = (kernel - 1) // 2

        self.start = nn.Linear(latent_dim, latent_dim * size_seq)
        self.lat_layer = nn.GRU(hidden_size, hidden_size,
                                batch_first=True, dropout=0,
                                bidirectional=True)
        self.gru_enc = nn.GRU(1 * 2, n_features,
                              batch_first=True, dropout=0,
                              bidirectional=False)
        self.layers_gru = nn.ModuleList([nn.GRU(
            32 * 2, 32, batch_first=True, bidirectional=False,
        ) for i in torch.arange(n_features)])
        self.last_layers_gru = nn.ModuleList([nn.GRU(
            1, 32, batch_first=True, bidirectional=True,
        ) for i in torch.arange(n_features)])

        self.cnns3 = nn.ModuleList([nn.ConvTranspose1d(
            in_channels=32,
            out_channels=64,
            padding=self.padd,
            kernel_size=(kernel,)) for i in torch.arange(n_features)])

        self.cnns2 = nn.ModuleList([nn.ConvTranspose1d(
            in_channels=64,
            out_channels=128,
            padding=self.padd,
            kernel_size=(kernel,)) for i in torch.arange(n_features)])

        self.cnns1 = nn.ModuleList([nn.ConvTranspose1d(
            in_channels=128,
            out_channels=1,
            padding=self.padd,
            kernel_size=(kernel,)) for i in torch.arange(n_features)])

        self.output_layer = nn.Linear(self.hidden_size * 2, 32, bias=True)
        self.leaky = nn.LeakyReLU()

    def __cnns(self, x):
        result = torch.empty(x.shape, device=x.device).transpose(1, 2)
        # print(result.shape)

        for i, cnn in enumerate(self.cnns3):
            input_cnn = x[:, :, i:i + 1]
            # print(input_cnn.shape)
            input_cnn, _ = self.last_layers_gru[i](input_cnn)
            # print(input_cnn.shape)
            input_cnn, _ = self.layers_gru[i](input_cnn)
            # print(input_cnn.shape)
            input_cnn = input_cnn.transpose(1, 2)
            # print(input_cnn.shape)/

            input_cnn = self.leaky(cnn(input_cnn))
            # print(input_cnn.shape)

            input_cnn = self.leaky(self.cnns2[i](input_cnn))
            # print(input_cnn.shape)
            # print(self.leaky(self.cnns1[i](input_cnn)).shape)
            result[:, i:i + 1, :] = self.leaky(self.cnns1[i](input_cnn))
        return result

    def forward(self, x):
        # x = self.start(x)
        # x = x.view(-1, self.size_seq, self.latent_dim)
        # x, _ = self.lat_layer(x)
        # print('encoder')
        # print(x.shape)

        x, _ = self.gru_enc(x.transpose(1, 2))
        # print(x.shape)
        # print(x.shape,_.shape)

        # x = x.transpose(1, 2)
        # x = self.__cnns(x)
        #   x = self.leaky(self.cnn_2(x))
        #    print(x.shape)
        #    x = self.leaky(self.cnn_1(x))
        # print(x.shape)
        return self.__cnns(x).transpose(1, 2)
