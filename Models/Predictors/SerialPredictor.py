from torch import nn
import torch
from Models.Predictors.BaseModel.Predictor import Predictor
from multiprocessing.dummy import Pool as ThreadPool


def fast_forward(data):
    predictor, x, reinit, init = data
    if reinit:
        predictor.init_hidden(x.shape[0])
    zeros = torch.full((x.shape[0],
                        predictor.size_subsequent,
                        x.shape[2]),
                       fill_value=init,
                       device=predictor.device)
    x = torch.cat([zeros, x], dim=1)
    number_batch = torch.isnan(x[:, :, :]).bool()
    number_batch_short = number_batch.any(axis=2)
    number_batch_short = number_batch_short.any(axis=0)

    for i in torch.arange(x.shape[1]):

        if number_batch_short[i]:

            sel_batch = number_batch[:, i, :].any(axis=1)
            batch = x[sel_batch, i - predictor.size_subsequent:i]
            batch[torch.isnan(batch)] = -1
            pred = predictor(batch)
            buf = x[sel_batch, i, :]
            index = torch.isnan(buf)
            buf[index] = pred[index]
            x[sel_batch, i, :] = buf

    x[torch.isnan(x)] = -1

    return x[:, predictor.size_subsequent:]


def forward(data):
    predictor, x, reinit, init = data
    if reinit:
        predictor.init_hidden(x.shape[0])
    zeros = torch.full((x.shape[0],
                        predictor.size_subsequent,
                        x.shape[2]),
                       fill_value=init,
                       device=predictor.device)
    x = torch.cat([zeros, x], dim=1)
    for i in range(predictor.size_subsequent, x.shape[1]):
        batch = x[:, i - predictor.size_subsequent:i]
        batch[torch.isnan(batch)] = -1

        if torch.isnan(x[:, i, :]).any():
            pred = predictor(batch)
            buf = x[:, i, :]
            index = torch.isnan(buf)
            buf[index] = pred[index]
            x[:, i, :] = buf
        else:
            with torch.no_grad():
                pred = predictor(batch)
    return x[:, predictor.size_subsequent:]


class SerialPredictor(nn.Module):
    def __init__(self,
                 forward_predictor: Predictor,
                 backward_predictor=None,
                 with_h=False,
                 init=-1):
        super().__init__()
        self.forward_predictor = forward_predictor
        self.backward_predictor = backward_predictor
        self.with_h = with_h
        self.thread = False

        self.init = init
        self.size_subsequent = self.forward_predictor.size_subsequent
        if with_h:
            self.forward_func = forward
        else:
            self.forward_func = fast_forward
        print(self.forward_func)

    def forward(self, x: torch.Tensor, reinit=True):
        # print(x.shape)
        if self.backward_predictor is not None:
            if self.thread:
                pool = ThreadPool(2)

                forw = [self.forward_predictor,
                        x,
                        self.init,
                        reinit]
                back = [self.backward_predictor,
                        torch.flip(x, [1]),
                        reinit,
                        self.init]
                f, b = pool.map(forward, [forw, back])
                b = torch.flip(b, [1])
                x = (f + b) / 2.0
            else:
                f = forward([self.forward_predictor,
                             x,
                             self.init,
                             reinit])
                b = forward([self.backward_predictor,
                             torch.flip(x, [1]),
                             reinit,
                             self.init])
                b = torch.flip(b, [1])
                #     if self.training:
                # ё            return f, b
                x = (f + b) / 2
        else:
            x = self.forward_func([self.forward_predictor,
                                   x,
                                   self.init,
                                   reinit])

        return x

    def fl(self):
        self.forward_predictor.fl()
