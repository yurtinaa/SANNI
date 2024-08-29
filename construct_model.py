from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import random
import logging

from .DataRead import PredictorDataset
from .Models.Predictors.ANNI.Predictor import ANPredictor
from .Models.Predictors.NotSerial import NotSerialPredictor
from .Models.Predictors.SAETI.model import SAETI
from .Models.Predictors.SAETI.model_with_snippet import ConstructSAETI
from .Models.Predictors.SerialPredictor import SerialPredictor
from .Models.Predictors.with_snippet import WithSnippetPredictor
from .Preprocess import get_not_nan_indices
from .Preprocess.snippet_construction import parallel_processing
from .model import SANNI


@dataclass
class SnippetModel(SANNI):
    predictor = None

    def _preproccess_with_classifier(self, data):
        train_results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            # print(batch.shape)
            batch_result = self.classifier(torch.Tensor(batch[:, :]).transpose(2, 1).to(self.device)) \
                .detach().cpu().numpy()
            batch_result = np.argmax(batch_result, axis=1)
            train_results.append(batch_result)
        train_results = np.concatenate(train_results, axis=0)
        train_snippet = []
        for dim in range(train_results.shape[1]):
            train_snippet_dim = parallel_processing(train_results[:, dim],
                                                    self.all_snippet[dim].cpu().numpy(),
                                                    data[:, :, dim],
                                                    self.size_subsequent,
                                                    500)
            train_snippet.append(train_snippet_dim)
        train_snippet = torch.Tensor(train_snippet).transpose(1, 2).transpose(0, 2)
        train_new = torch.cat((torch.Tensor(data), train_snippet), dim=2)
        train_new[:, :, ::2] = torch.Tensor(data)
        train_new[:, :, 1::2] = torch.Tensor(train_snippet)
        return train_new

    def _create_dataset(self, train, valid):
        if self.check_nan:
            logging.debug('проверил на nan')
            not_nan_index = get_not_nan_indices(train)
            train = train[not_nan_index]
            logging.debug(str(train.shape))
            logging.debug(str(np.isnan(train).any()))
            not_nan_index = get_not_nan_indices(valid)
            valid = valid[not_nan_index]
        else:
            logging.debug('не проверял на nan')
        train = self._preproccess_with_classifier(train)
        valid = self._preproccess_with_classifier(valid)
        self.pr_dataset = PredictorDataset(arr_train=train,
                                           arr_val=valid,
                                           type_nan='None',
                                           batch_size=self.batch_size,
                                           size_subsequent=self.size_subsequent,
                                           seed=self.seed,
                                           device=self.device)

    def preprocess_data_to_predictor(self, train, valid):
        self._create_dataset(train, valid)
        if self.all_snippet is not None:
            self.all_snippet = self.all_snippet.to(self.device)
            logging.debug(f"Snippet shape: {self.all_snippet.shape}")
        if self.model_predictor == 'saeti':
            self.predictor = ConstructSAETI(size_seq=self.size_subsequent,
                                            n_features=self.all_snippet.shape[0],
                                            hidden_size=self.size_subsequent,
                                            latent_dim=self.size_subsequent // 2,
                                            classifier=None,
                                            snippet_list=self.all_snippet)
            # model_log_func(self.predictor)
        else:
            predictor = self._init_sanni_predictor()
            predictor.classifier = None
            predictor.with_h = self.with_h
            backward_predictor = None
            self.predictor = WithSnippetPredictor(predictor)
            logging.debug(str(self.model_predictor))

        self.predictor = self.predictor.to(self.device)

    def _impute_preprocessing(self, x):
        miss = torch.Tensor(x).to(self.device)
        #     print(miss.shape).
        result = []
        for i in range(0, len(miss), self.batch_size):
            batch = miss[i:i + self.batch_size]
            # print('batch:', batch.shape)
            # print(batch.shape)
            # batch[torch.isnan(batch)] = 0.
            batch_result = self.classifier(torch.Tensor(batch[:, :]).transpose(2, 1).to(self.device)) \
                .detach().cpu().numpy()
            batch_result = np.argmax(batch_result, axis=1)

            result.append(batch_result)
        result = np.concatenate(result, axis=0)
        result_snippet = []
        for dim in range(result.shape[1]):
            train_snippet_dim = parallel_processing(result[:, dim],
                                                    self.all_snippet[dim].cpu().numpy(),
                                                    miss[:, :, dim].cpu().numpy(),
                                                    self.size_subsequent,
                                                    500)
            result_snippet.append(train_snippet_dim)
        result_snippet = torch.Tensor(result_snippet).transpose(1, 2).transpose(0, 2).to(self.device)
        miss_new = torch.cat((torch.Tensor(miss), result_snippet), dim=2)
        miss_new[:, :, ::2] = torch.Tensor(miss)
        miss_new[:, :, 1::2] = torch.Tensor(result_snippet)
        return miss_new

    def impute(self, x):
        # print(x.shape)
        miss = self._impute_preprocessing(x)
        # print(miss.shape)

        result_predict = []
        for i in range(0, len(miss), self.batch_size):
            batch = miss[i:i + self.batch_size]

            batch_result = self.predictor(batch)
            result_predict.append(batch_result.detach().cpu().numpy(

            ))

        result = np.concatenate(result_predict, axis=0)
        return result[:, :, ::2]
