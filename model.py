from dataclasses import dataclass
from pathlib import Path
import sys
import logging
# sys.path.append(Path('..') / '..' / '..')
import torch
import numpy as np
import random
from datetime import datetime
from .Models import Predictor, Classifier, SAETI
from .Models.Predictors.BaseModel.ShortPredictor import ShortPredictor
from .Models.Predictors.ANNI.Predictor import ANPredictor
from .Models.Predictors.SerialPredictor import SerialPredictor
from .Models.Predictors.NotSerial import NotSerialPredictor
from .Preprocess import get_snippets, get_not_nan_indices
from .DataRead import ClassifierDataset, PredictorDataset
from .Utils import Trainer, classifier_score, check_model_classifier
from utils.logs import log_func


def model_log_func(*args):
    log_func(*args, level='sanni_model')


mse = torch.nn.MSELoss()


def func_saeti_loss(x, y, pred):
    # index = x != x
    index = y != y

    # index[index_origin] = False
    return mse(y[~index], pred[~index])


def func_mse_loss(x, y, pred):
    index = x != x
    index_origin = y != y

    index[index_origin] = False
    return mse(y[index], pred[index])


def rolling_window(a, window):
    # result = torch.zeros(size=(a.shape[0],a.shape[1]))
    returns = []
    for i in np.arange(0, a.shape[-2], window):
        returns.append(a[:, i:i + window, :])
    return torch.cat(returns)


def set_seed(seed):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


models_predictor = {
    "last_point": Predictor,
    'short_serial': ShortPredictor,
    "anni_last": ANPredictor,
    'saeti': SAETI,
}


# fixme: добавить обработку исключени:
# 1. Остановки по кнопке
# 2. Остановки из за большой выборки.

def init_nan(data):
    return data


def init_zeros(data):
    data[data != data] = 0.0
    return data


def init_mean(data):
    for col_inx in np.arange(data.shape[-1]):
        col_data = data[:, :, col_inx]
        col_data[col_data != col_data] = np.nanmean(col_data)
        data[:, :, col_inx] = col_data
    return data


key_init_snippet_func = {
    'zeros': init_zeros,
    'nan': init_nan,
    'mean': init_mean,

}

key_loss_model_func = {
    'saeti': func_saeti_loss,
    "last_point": func_mse_loss,
    'anni_last': func_mse_loss,
    'short_serial': func_mse_loss,
}


@dataclass
class SANNI:
    size_subsequent: int
    count_snippet: int
    batch_size: int = 128
    backward: bool = False

    num_layers: int = 1
    hidden: int = 128
    cell: str = 'gru'
    kernel: int = 5
    augmentation: int = 0
    model_predictor: str = "last_point"
    type_nan: str = "last"
    device: torch.device = torch.device('cpu')
    bar: bool = False
    log_train: bool = False
    sh: bool = False
    with_h: bool = False
    epoch_pr: int = 1000
    epoch_cl: int = 200
    init_snippet: str = 'nan'
    dim: int = None
    add_nan: str = 'batch'
    lr_classifier: float = 1.0e-3
    lr_predict: float = 1.0e-3
    inside_count: int = 5
    seed: int = 3515
    serial: bool = True
    save_dir: Path = None
    snippet_size: int = None
    all_snippet = None

    check_nan: bool = True
    wandb_log = False
    classifier: Classifier = None

    def __post_init__(self):
        set_seed(self.seed)
        assert self.model_predictor in models_predictor.keys()
        if self.init_snippet not in key_init_snippet_func:
            raise ValueError('not correct init_snippet:(nan, zeros)')
        logging.info(f'model: {self.model_predictor}')
        if self.snippet_size is None:
            self.snippet_size = self.size_subsequent // 2

        self.pred_func = key_loss_model_func[self.model_predictor]

    def preprocess_data_to_classifier(self, train):
        if len(train.shape) > 2:
            self.dim = train.shape[2]
        else:
            self.dim = 1
        # print('count_snippet', self.count_snippet)
        train = train.copy()
        logging.info(f'init snippet:{self.init_snippet}')
        train = key_init_snippet_func[self.init_snippet](train)
        logging.info(f'Размерность обучающей выборки classifier:{train.shape}')
        train = rolling_window(torch.Tensor(train), self.snippet_size).numpy()
        logging.debug(f"Train and snippet size: {train.shape, self.snippet_size}")
        #   valid = rolling_window(torch.Tensor(train), self.snippet_size)
        train_not_nan, snippet_list, not_nan_index = get_snippets(train,
                                                                  self.count_snippet)

        logging.info(f'Сlassifier train set size{train_not_nan.shape}')
        if train_not_nan.shape[0] < self.batch_size:
            raise ValueError('Сlassifier train set size is smaller than the batch size.')

        self.all_snippet = []
        for arr_snippet in snippet_list:
            snippets = []
            for snippet in arr_snippet:
                snippets.append(snippet['snippet'])
            self.all_snippet.append(snippets)
        self.all_snippet = torch.Tensor(self.all_snippet)
        self.count_snippet = self.all_snippet.shape[1]
        if self.model_predictor == 'saeti':
            snippet_size = self.snippet_size + 1
            all_data = 'all'
        else:
            snippet_size = self.snippet_size
            all_data = 'none'
        self.classifier = Classifier(size_subsequent=snippet_size,
                                     count_snippet=self.count_snippet,
                                     dim=self.dim)
        self.classifier = self.classifier.to(self.device)
        logging.debug(str(self.classifier))
        self.cl_dataset = ClassifierDataset(arr_data=train_not_nan,
                                            snippet_list_arr=snippet_list,
                                            batch_size=64,
                                            size_subsequent=snippet_size,
                                            seed=self.seed,
                                            all_data=all_data,
                                            device=self.device)

    def _init_sanni_predictor(self, backward=False) -> Predictor:
        return models_predictor[self.model_predictor](self.snippet_size,
                                                      count_snippet=self.count_snippet,
                                                      num_layers=self.num_layers,
                                                      classifier=self.classifier,
                                                      snippet_list=self.all_snippet,
                                                      hidden_dim=self.hidden,
                                                      backward=backward,
                                                      dim=self.dim,
                                                      inside_count=0,
                                                      kernel=self.kernel,
                                                      device=self.device,
                                                      cell=self.cell,
                                                      )

    def preprocess_data_to_predictor(self, train, valid):
        if self.check_nan:
            logging.debug('проверил на nan')
            not_nan_index = get_not_nan_indices(train)
            train = train[not_nan_index]
            logging.debug(str(train.shape))
            logging.debug(str(np.isnan(train).any()))
            not_nan_index = get_not_nan_indices(valid)
            valid = valid[not_nan_index]
        else:
            logging.debug('dont check nan')

        self.pr_dataset = PredictorDataset(arr_train=train,
                                           arr_val=valid,
                                           type_nan='None',
                                           batch_size=self.batch_size,
                                           size_subsequent=self.size_subsequent,
                                           seed=self.seed,
                                           device=self.device)
        if self.all_snippet is not None:
            self.all_snippet = self.all_snippet.to(self.device)
            logging.debug(f"Snippet shape: {self.all_snippet.shape}")
        if self.model_predictor == 'saeti':
            self.predictor = SAETI(size_seq=self.size_subsequent,
                                   n_features=self.all_snippet.shape[0],
                                   hidden_size=self.size_subsequent,
                                   latent_dim=self.size_subsequent // 2,
                                   classifier=self.classifier,
                                   snippet_list=self.all_snippet)
            # model_log_func(self.predictor)
        else:
            predictor = self._init_sanni_predictor()

            predictor.with_h = self.with_h
            backward_predictor = None
            if self.serial:
                if self.backward:
                    backward_predictor = self._init_sanni_predictor(backward=True)
                    backward_predictor.with_h = self.with_h
                    backward_predictor = backward_predictor.to(self.device)

                self.predictor = SerialPredictor(forward_predictor=predictor,
                                                 backward_predictor=backward_predictor)

            else:
                self.predictor = NotSerialPredictor(predictor)
                logging.debug(str(self.model_predictor))

        self.predictor = self.predictor.to(self.device)

        logging.debug(str(self.predictor))

    def train_predictor(self):
        optimizer = torch.optim.Adam(self.predictor.parameters(),
                                     lr=self.lr_predict, amsgrad=True)
        scheduler = None
        if self.sh:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   patience=20)
        self.predictor.train()
        trainer = Trainer(model=self.predictor,
                          optimizer=optimizer,
                          epochs_count=self.epoch_pr,
                          loss=self.pred_func,
                          dataset=self.pr_dataset,
                          scheduler=scheduler,
                          check_out=lambda x: False,
                          add_nan=self.add_nan,
                          score_func=self.pred_func,
                          name_log='predictor',
                          device=self.device)
        trainer.train()
        self.predictor = trainer.best_model
        #   self.predictor.fl()
        self.predictor.eval()

    def train_classifier(self):
        self.classifier.train()
        loss = torch.nn.CrossEntropyLoss()

        def cross(x, y, pred):
            # print(y)
            return loss(pred, y)

        def score(x, y, pred):
            return classifier_score(true=y.cpu(), predict=pred.cpu())

        optimizer = torch.optim.Adam(self.classifier.parameters(),
                                     lr=self.lr_classifier)
        trainer = Trainer(model=self.classifier,
                          optimizer=optimizer,
                          loss=cross,
                          epochs_count=self.epoch_cl,
                          dataset=self.cl_dataset,
                          score_func=score,
                          name_log='classifier',
                          add_nan=self.add_nan,
                          device=self.device,
                          check_out=check_model_classifier)
        trainer.train()
        self.classifer = trainer.best_model
        self.classifer.eval()
        # self.best_epoch_i, trainer.best_val_loss

    def fit(self, train, valid):
        logging.debug(f'hidden: {self.hidden}')
        logging.info('старт обучения')
        self.preprocess_data_to_classifier(train)
        logging.info('предобработка данных')

        self.train_classifier()
        logging.info('обучил классификатора')
        self.preprocess_data_to_predictor(train, valid)
        #   print('with_h: ', self.predictor.forward_predictor.with_h)
        logging.info('обучаю предсказателя')
        self.train_predictor()

    def impute(self, x):
        miss = torch.Tensor(x).to(self.device)
        #     print(miss.shape)
        result_predict = []
        for i in range(0, len(miss), self.batch_size):
            batch = miss[i:i + self.batch_size]

            batch_result = self.predictor(batch)
            result_predict.append(batch_result.detach().cpu().numpy(

            ))
            # print(np.isnan(result_predict).any())

            # predict = self.predictor(miss[:, :])
        result = np.concatenate(result_predict, axis=0)

        # miss[:, :] = predict[:, :]
        # print(np.isnan(miss.detach
        # ().cpu().numpy()).any())
        return result

    def change_snippet(self, type_):
        # rint('изменил свои сниппеты')
        if self.model_predictor != 'saeti':
            self.predictor.forward_predictor.change_snippet(type_)
        elif self.model_predictor != 'anni_last':
            self.predictor.change_snippet(type_)
