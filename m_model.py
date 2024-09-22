# recovery[None,i-window+1:i+1]# from lib.SANNI.Models.Predictors.NoPoint import NoPoint
# from .Utils.testing_methods import classifier_score
from .model import *
from .Models.Predictors.Multivariate.Base import BaseMultivarite


# zxczxc
# fixme: добавить обработку исключени:
# 1. Остановки по кнопке
# 2. Остановки из за большой выборки.

@dataclass
class MSANNI(SANNI):
    classifier_list = []
    cl_dataset_list = []

    def preprocess_data_to_classifier(self, train):
        if len(train.shape) > 2:
            self.dim = train.shape[2]
        else:
            self.dim = 1
        #  print('count_snippet', self.count_snippet)
        train = rolling_window(torch.Tensor(train), self.snippet_size).numpy()
        #   valid = rolling_window(torch.Tensor(train), self.snippet_size)
        train_not_nan, snippet_list, not_nan_index = get_snippets(train,
                                                                  self.count_snippet)
        self.all_snippet = []
        for arr_snippet in snippet_list:
            snippets = []
            for snippet in arr_snippet:
                snippets.append(snippet['snippet'])
            self.all_snippet.append(snippets)
        self.all_snippet = torch.Tensor(self.all_snippet)
        self.count_snippet = self.all_snippet.shape[1]
     #   print(train.shape)
        for dim in np.arange(self.dim):
            classifier = Classifier(size_subsequent=self.snippet_size,
                                    count_snippet=self.count_snippet,
                                    dim=1)
            classifier = classifier.to(self.device)
            self.classifier_list.append(classifier)
            cl_dataset = ClassifierDataset(arr_data=train_not_nan[:, :, dim:dim + 1],
                                           snippet_list_arr=snippet_list[dim:dim + 1],
                                           batch_size=self.batch_size,
                                           size_subsequent=self.snippet_size,
                                           seed=self.seed,
                                           device=self.device)
            self.cl_dataset_list.append(cl_dataset)

    def preprocess_data_to_predictor(self, train, valid):
        # fixme ПОЧЕМУ БЕЗ НАН
        not_nan_index = get_not_nan_indices(train)
        train = train[not_nan_index]
        not_nan_index = get_not_nan_indices(valid)
        valid = valid[not_nan_index]

        self.pr_dataset = PredictorDataset(arr_train=train,
                                           arr_val=valid,
                                           batch_size=self.batch_size,
                                           size_subsequent=self.size_subsequent,
                                           seed=self.seed,
                                           device=self.device)
        predictors = []
        for dim in np.arange(self.dim):
           # print(self.classifier_list)
            predictor = models_predictor[self.model_predictor](self.snippet_size,
                                                               self.count_snippet,
                                                               num_layers=self.num_layers,
                                                               classifier=self.classifier_list[dim],
                                                               snippet_list=self.all_snippet,
                                                               hidden_dim=self.hidden,
                                                               dim=1,
                                                               inside_count=0,
                                                               kernel=self.kernel,
                                                               device=self.device,
                                                               cell=self.cell,
                                                               )
            predictor.with_h = self.with_h
            predictor = predictor.to(self.device)
            backward_predictor = None
            if self.serial:
                if self.backward:
                    backward_predictor = models_predictor[self.model_predictor](self.snippet_size,
                                                                                self.count_snippet,
                                                                                num_layers=self.num_layers,
                                                                                classifier=self.classifier,
                                                                                snippet_list=torch.flip(
                                                                                    self.all_snippet,
                                                                                    [2]),
                                                                                hidden_dim=self.hidden,
                                                                                dim=self.dim,
                                                                                inside_count=0,
                                                                                backward=True,
                                                                                kernel=self.kernel,
                                                                                device=self.device,
                                                                                cell=self.cell,
                                                                                )
                    backward_predictor.with_h = self.with_h
                    backward_predictor = backward_predictor.to(self.device)

                predictor = SerialPredictor(forward_predictor=predictor,
                                            backward_predictor=backward_predictor)
            predictors.append(predictor)
        self.predictor = BaseMultivarite(torch.nn.ModuleList(predictors),
                                         dim=self.dim)
        self.predictor = self.predictor.to(self.device)

    '''
    def train_predictor(self):
        optimizer = torch.optim.Adam(self.predictor.parameters(),
                                     lr=self.lr_predict, amsgrad=True)
        self.predictor.train()
        trainer = Trainer(model=self.predictor,
                          optimizer=optimizer,
                          epochs_count=self.epoch_pr,
                          loss=func_mse_loss,
                          dataset=self.pr_dataset,
                          score_func=func_mse_loss,
                          name_log='predictor',
                          device=torch.device('cuda'),
                          check_out=lambda x: False)
        trainer.train()
        self.predictor = trainer.best_model
        self.predictor.fl()
        self.predictor.eval()
    '''

    def train_classifier(self):
        for dim, classifier in enumerate(self.classifier_list):
            classifier.train()
            loss = torch.nn.CrossEntropyLoss()

            def cross(x, y, pred):
                # print(y)
                return loss(pred, y)

            def score(x, y, pred):
                return classifier_score(y.cpu(), pred.cpu())

            optimizer = torch.optim.Adam(classifier.parameters(),
                                         lr=self.lr_classifier)
            trainer = Trainer(model=classifier,
                              optimizer=optimizer,
                              loss=cross,
                              epochs_count=self.epoch_cl,
                              dataset=self.cl_dataset_list[dim],
                              score_func=score,
                              name_log='classifier',
                              device=torch.device('cuda'),
                              check_out=check_model_classifier)
            trainer.train()
            classifer = trainer.best_model
            classifer.eval()
            self.classifier_list[dim] = classifier
        # self.best_epoch_i, trainer.best_val_loss
