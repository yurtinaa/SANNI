import torch
import torch.nn as nn
from pygrinder import fill_and_get_mask_torch
from pypots.utils.metrics import calc_mae

from AbstractModel.error.AbstractError import AbstractError
from AbstractModel.error.TorchError import TorchImputeError, BaseErrorTorch
from Trainer.AbstractTrainer import AbstractModel
from .backbone import BackboneBRITS
import torch
import numpy as np

def calculate_mae(predictions, targets):
    """
    Вычисляет среднее абсолютное отклонение (MAE) между предсказаниями и целевыми значениями.

    :param predictions: Тензор с предсказаниями.
    :param targets: Тензор с истинными значениями.
    :return: Среднее абсолютное отклонение (MAE).
    """
    absolute_errors = torch.abs(predictions - targets)
    mae = absolute_errors.mean()
    return mae


def reverse_tensor(tensor_):
    if tensor_.dim() <= 1:
        return tensor_
    indices = range(tensor_.size()[1])[::-1]
    indices = torch.tensor(
        indices, dtype=torch.long, device=tensor_.device, requires_grad=False
    )
    return tensor_.index_select(1, indices)


class BritsTorchError(BaseErrorTorch):
    def _get_consistency_loss(self, pred_f: torch.Tensor, pred_b: torch.Tensor) -> torch.Tensor:
        loss = self.loss(pred_f, pred_f, pred_b).mean() * 1e-1
        return loss

    def __call__(self, X, Y, Y_pred):
        missing_mask = X['forward']["missing_mask"].bool()
        forward_Y, _ = fill_and_get_mask_torch(Y)
        X = X['forward']["X"]
        index_mask = missing_mask.bool()
        h_dict = Y_pred['h_dict']
        forward_loss = torch.tensor(0.0).to(Y.device)
        loss_forward_loss = torch.tensor(0.0).to(Y.device)
        for key, predict in h_dict['forward'].items():
            index = X != X
            index_origin = Y != Y

            index[index_origin] = False
            # loss_forward_loss += calc_mae(predict, forward_Y, missing_mask)
            # print(Y[missing_mask.bool()].shape)
            forward_loss += self.loss(X, Y, predict).mean()
        backward_loss = torch.tensor(0.0).to(Y.device)
        for key, predict in h_dict['backward'].items():
            predict_reverse = reverse_tensor(predict)
            index = X != X
            index_origin = Y != Y

            index[index_origin] = False
            # loss_forward_loss += calc_mae(predict_reverse, forward_Y, missing_mask)

            backward_loss += self.loss(X, Y, predict_reverse).mean()
        reconstruction_loss = (forward_loss + backward_loss) / 3
        consistency_loss = self._get_consistency_loss(Y_pred['f_imputed_data'],
                                                      Y_pred['b_imputed_data'])
        # print('f_my', loss_forward_loss / 3, 'b_my', reconstruction_loss)
        return consistency_loss + reconstruction_loss


class _BRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    """

    def __init__(
            self,
            n_steps: int,
            n_features: int,
            rnn_hidden_size: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.model = BackboneBRITS(n_steps, n_features, rnn_hidden_size)

    @property
    def device(self):
        first_param = next(self.model.parameters(), None)
        return first_param.device

    def forward(self, inputs: dict, training: bool = True) -> dict:
        (
            imputed_data,
            f_imputed_data,
            b_imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
            h_dict
        ) = self.model(inputs)

        results = {
            "imputed_data": imputed_data,
            "f_imputed_data": f_imputed_data,
            "b_imputed_data": b_imputed_data,
            "h_dict": h_dict,
        }
        loss_function = BritsTorchError(nn.L1Loss())
        # my_loss = loss_function(inputs, inputs['forward']["X"], results)
        # if in training mode, return results with losses
        if training:
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            loss = consistency_loss + reconstruction_loss
            # print(loss)
            # `loss` is always the item for backward propagating to update the model
            # results["loss"] = my_loss
            results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction

        return results
