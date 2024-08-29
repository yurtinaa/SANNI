import torch
from lib.SANNI.Models.Helpers import Helpers
import json
from lib.SANNI.Preprocess.const import *
from pathlib import Path
import numpy as np


def train_model(size_subsequent,
                dataset_path,
                count_snippet,
                learning_rate,
                sh=False,
                batch_size=128,
                sh_pat=150,
                sh_fact=0.95,
                inside=3,
                wandb=None,
                model='hard',
                bar=False,
                kernel=5,
                opt='adam',
                epochs=1500,
                epoch_cl=1000,
                save=False,
                seed=3515,
                device=None,
                hidden=0):
    if hidden == 0:
        hidden = size_subsequent
    if device is None:
        device = torch.device('cuda')
    if save:
        result = dataset_path / RESULT_DIR

        if not result.is_dir():
            result.mkdir()
        save_path = result / f"{np.random.randint(10000, 99999)}"
        while save_path.is_dir():
            save_path = result / f"{np.random.randint(10000, 99999)}"
            print(save_path.name)
        save_path.mkdir()
        print(save_path)
    else:
        save_path = None
    helpers_model = Helpers(size_subsequent=size_subsequent,
                            dataset=dataset_path,
                            batch_size=batch_size,
                            device=device,
                            bar=bar,
                            log_train=False,
                            wandb=wandb,
                            kernel=kernel,
                            inside_count=inside,
                            save_dir=save_path,
                            seed=seed,
                            count_snippet=count_snippet,
                            model_predictor=model,
                            hidden=hidden)
    print(helpers_model.device)
    # helpers_model.preprocessing_data()
    helpers_model.train_classifier(epoch_cl=epoch_cl)
    torch.cuda.empty_cache()
    helpers_model.test_classifier(False)
    torch.cuda.empty_cache()
    helpers_model.train_predictor(epoch_pr=epochs,
                                  lr=learning_rate,
                                  sh=sh,
                                  opt=opt,
                                  pat=sh_pat,
                                  fact=sh_fact)
    #scen = helpers_model.test_scenario()
    return None, helpers_model


def init_model_from_path(dataset: Path,
                         id_result):
    result_path = dataset / RESULT_DIR / id_result
    config_file = result_path / MODEL_CONFIG_FILE
    with open(config_file, 'r+') as config_file:
        config_file = json.load(config_file)
    helpers_model = Helpers(size_subsequent=config_file['size_subsequent'],
                            dataset=dataset,
                            seed=config_file['seed'],
                            num_layers=config_file['num_layers'],
                            cell=config_file['cell'],
                            device=torch.device('cuda'),
                            kernel=config_file['kernel'],
                            inside_count=config_file['inside'],
                            count_snippet=config_file['count_snippet'],
                            model_predictor=config_file['model_predictor'],
                            save_dir=result_path,
                            hidden=config_file['hidden'])
    # helpers_model.preprocessing_data()
    helpers_model.open_model(open_classifer=True)

    return helpers_model
