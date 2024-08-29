import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from .construct_model import SnippetModel
from utils.config.model_base_config import saits_base_config, model_config_base

import numpy as np

from utils.config.base_model_init import base_model_init




@dataclass
class SnippetOtherModel(SnippetModel):
    def __post_init__(self):
        # assert self.model_predictor in models_predictor.keys()
        # if self.init_snippet not in key_init_snippet_func:
        #     raise ValueError('not correct init_snippet:(nan, zeros)')
        logging.info(f'model: {self.model_predictor}')
        if self.snippet_size is None:
            self.snippet_size = self.size_subsequent // 2

        # model_log_func('model_predict', self.model_predictor)
        # self.pred_func = key_loss_model_func[self.model_predictor]

    def preprocess_data_to_predictor(self, train, valid):
        # if self.check_nan:
        #     logging.debug('проверил на nan')
        #     not_nan_index = get_not_nan_indices(train)
        #     train = train[not_nan_index]
        #     logging.debug(str(train.shape))
        #     logging.debug(str(np.isnan(train).any()))
        #     not_nan_index = get_not_nan_indices(valid)
        #     valid = valid[not_nan_index]
        # else:
        #     logging.debug('не проверял на nan')
        # # train_results = []
        # # for i in range(0, len(train), self.batch_size):
        # #     batch = train[i:i + self.batch_size]
        # #     batch_result = self.classifier(torch.Tensor(batch[:, :-1]).transpose(2, 1).to(self.device)) \
        # #         .detach().cpu().numpy()
        # #     batch_result = np.argmax(batch_result, axis=1)
        # #
        # #     train_results.append(batch_result)
        # # train_results = np.concatenate(train_results, axis=0)
        # # train_snippet = []
        # # for dim in range(train_results.shape[1]):
        # #     train_snippet_dim = parallel_processing(train_results[:, dim],
        # #                                             self.all_snippet[dim].cpu().numpy(),
        # #                                             train[:, :, dim],
        # #                                             self.size_subsequent,
        # #                                             500)
        # #     train_snippet.append(train_snippet_dim)
        # # train_snippet = torch.Tensor(train_snippet).transpose(1, 2).transpose(0, 2)
        # # train_new = torch.cat((torch.Tensor(train), train_snippet), dim=2)
        # # train_new[:, :, ::2] = torch.Tensor(train)
        # # train_new[:, :, 1::2] = torch.Tensor(train_snippet)
        # train = self._preproccess_with_classifier(train)
        # valid = self._preproccess_with_classifier(valid)
        #
        # # test_results = []
        # # for i in range(0, len(valid), self.batch_size):
        # #     batch = valid[i:i + self.batch_size]
        # #     batch_result = self.classifier(torch.Tensor(batch[:, :-1]).transpose(2, 1).to(self.device)) \
        # #         .detach().cpu().numpy()
        # #     batch_result = np.argmax(batch_result, axis=1)
        # #
        # #     test_results.append(batch_result)
        # # test_results = np.concatenate(test_results, axis=0)
        # #
        # # valid_snippet = []
        # # for dim in range(train_results.shape[1]):
        # #     valid_snippet_dim = parallel_processing(test_results[:, dim],
        # #                                             self.all_snippet[dim].cpu().numpy(),
        # #                                             valid[:, :, dim],
        # #                                             self.size_subsequent,
        # #                                             500)
        # #     valid_snippet.append(valid_snippet_dim)
        # # valid_snippet = torch.Tensor(valid_snippet).transpose(1, 2).transpose(0, 2)
        # # valid_new = torch.cat((torch.Tensor(valid), valid_snippet), dim=2)
        # # valid_new[:, :, ::2] = torch.Tensor(valid)
        # # valid_new[:, :, 1::2] = torch.Tensor(valid_snippet)
        #
        # # print(train_results.shape, train.shape, valid_snippet.shape)
        # self.pr_dataset = PredictorDataset(arr_train=train,
        #                                    arr_val=valid,
        #                                    type_nan='None',
        #                                    batch_size=self.batch_size,
        #                                    size_subsequent=self.size_subsequent,
        #                                    seed=self.seed,
        #                                    device=self.device)
        self._create_dataset(train, valid)
        config = model_config_base[self.model_predictor]
        config['n_steps'] = self.size_subsequent
        config['n_features'] = self.dim * 2
        config['batch_size'] = self.batch_size
        config['real_device'] = self.device
        self.predictor = base_model_init[self.model_predictor](config)
        # self.predictor = SAITS(n_steps=self.size_subsequent,
        # n_features=self.dim * 2,
        # )

    def train_predictor(self):
        train = self.pr_dataset.dataset['train'][0]
        val = self.pr_dataset.dataset['val'][0]
        self.predictor.fit(train, val)

    def impute(self, x):
        miss = self._impute_preprocessing(x)
        miss[:, -1, ::2]=np.nan
        # print(miss[:, -1])
        result_predict = []
        for i in range(0, len(miss), self.batch_size):
            batch = miss[i:i + self.batch_size]

            batch_result = self.predictor.impute(batch)
            result_predict.append(batch_result)

        result = np.concatenate(result_predict, axis=0)
        return result[:, :, ::2]
