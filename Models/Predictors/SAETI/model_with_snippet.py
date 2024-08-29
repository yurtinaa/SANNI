from .model import SAETI


class ConstructSAETI(SAETI):
    def forward(self, x):
        x = x.clone()
        pred_x = x.clone()
        data = pred_x[:, :, ::2]
        snippet = pred_x[:, :, 1::2]
        # print(data.shape,snippet.shape)
        index = data != data
        data[index] = snippet[index]
        lat = self.encoder(data, snippet.transpose(2,1))
        x = self.decoder(lat)
        pred_x[:, :, ::2] = x
        return pred_x
