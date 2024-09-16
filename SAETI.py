from dataclasses import dataclass
from Models.Predictors.SAETI.model import SAETI as BaseSAETI
import torch

from SANNI import SANNI


@dataclass
class SAETI(SANNI):

    def _predictor_construct(self):
        predictor = BaseSAETI(size_seq=self.time_series.window_size,
                              n_features=self.time_series.dim,
                              hidden_size=self.time_series.window_size,
                              latent_dim=self.time_series.window_size // 2,
                              classifier=self._classifier,
                              snippet_list=torch.tensor(self._snippet_array).to(self.device))
        return predictor.to(self.device)

    def __post_init__(self):
        if self.name is None:
            self.name = 'SAETI'
        super().__post_init__()
