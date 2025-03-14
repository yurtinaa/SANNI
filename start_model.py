import argparse
import json
import string
import random
import traceback
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv
import os

from .AbstractModel.Parametrs import TimeSeriesConfig, TorchNNConfig
from .AbstractModel.error.TorchError import get_error
from .AbstractModel.optimizer.abstract_optimizer import Adam
from .AbstractModel.score import get_score, ScoreType
from .DataAnalyze.DataAnnotation import SnippetAnnotation
from .DataProducers.Convertors import SliceTimeSeriesConvertor, DropMissingSubConvertor
from .DataProducers.ImputeScenario import BlackoutScenario
from .DataProducers.ModelsBehavior.AbstractBehavior import SerialImputeBehavior
from .DataProducers.Normalizers import StandardNormalizer

from .DataProducers.Normalizers import MinMaxNormalizer
# from Logger.ConsoleLogger import ConsoleLogger
from .Logger.FileLogger import FileLogger
from .ModelsList import get_model

# Загружаем переменные из файла .env
load_dotenv()

# Читаем переменные
TOKEN_BOT = os.getenv("TOKEN_BOT")
CHAT_ID_TOKEN = os.getenv("CHAT_ID_TOKEN")


def token_generate():
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return random_string


def telegram_send(*args, **kwargs):
    message = ""
    if 'config' in kwargs:
        for key, value in kwargs['config'].items():
            message += f"{key}: {value}\n"
    message += f"message: {', '.join([str(x) for x in args])}"
    url = f"https://api.telegram.org/bot{TOKEN_BOT}/" \
          f"sendMessage?chat_id={CHAT_ID_TOKEN}&text={message}"
    requests.get(url)  #


def set_seed(seed):
    # Устанавливаем seed для Python
    random.seed(seed)

    # Устанавливаем seed для NumPy
    np.random.seed(seed)

    # Устанавливаем seed для PyTorch (CPU)
    torch.manual_seed(seed)

    # Если вы используете GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Для всех устройств (если несколько GPU)

    # Гарантируем, что поведение некоторых операций, например, вызовов cuDNN, будет детерминированным
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def start_train(config):
    try:
        torch.set_default_dtype(torch.float32)
        print(config)
        set_seed(34124)
        names = [path.name[:-9] for path in Path('result').iterdir()]

        dataset_name = config.get("dataset")

        current_directory = Path.cwd()

        # Список всех файлов и папок
        contents = list(current_directory.iterdir())

        print(contents)

        dataset_origin = np.loadtxt(f'{config.get("dataset")}.txt')
        dataset_blackout = BlackoutScenario().convert(dataset_origin)
        normalizer = StandardNormalizer()
        data_normalize = normalizer.fit(dataset_blackout)
        window_size = config.get('windows', 100)
        dataset_slice = SliceTimeSeriesConvertor(window_size).convert(data_normalize)
        drop_nan_sub = DropMissingSubConvertor().convert(dataset_slice)
        dataset_origin_norm = normalizer(dataset_origin)

        model_config = config.get('model').copy()

        error_config = config.get('error', {'type': 'MSE'})
        error = get_error(error_config['type'])(**error_config['params'])
        config['token'] = token_generate()

    except BaseException as e:
        full_traceback = traceback.format_exc()
        config_as_str = '\n'.join([f'{key}: {value}' for key, value in config.items()])
        telegram_send("Error init: \n" + config_as_str +
                      '\n' + str(full_traceback))
        raise

    try:
        config_as_str = '\n'.join([f'{key}: {value}' for key, value in config.items()])
        telegram_send("Start \n" + config_as_str)

        if 'snippet_count' in model_config:
            snippet_dict, train_set = SnippetAnnotation(config.get('fragment',
                                                                   window_size // 2),
                                                        config.get('snippet_count', 2)).annotate(drop_nan_sub)
            model_config['snippet_dict'] = snippet_dict
            model_config['snippet_list'] = train_set
        time_series_config = TimeSeriesConfig(drop_nan_sub.shape[2],
                                              drop_nan_sub.shape[1])

        train_config = TorchNNConfig(
            batch_size=config.get('batch_size', 64),
            epochs=config.get('epochs', 500),
            error_factory=error,
            optimizer_type=Adam(config['lr']),
            score_factory=get_score(ScoreType.MSE),
            early_stopping_patience=50
        )
        name = f'{model_config["name"]}_' + "_" + dataset_name.split('/')[-1] + str(error)
        name += '_' + token_generate()

        config['save_dir'] = name
        save_dir = Path('result') / name
        save_dir.mkdir(exist_ok=True,
                       parents=True)
        print_logger = FileLogger().configure(log_file=save_dir / 'train.log')
        model_config['time_series'] = time_series_config
        model_config['neural_network_config'] = train_config
        model_config['logger'] = print_logger
        model_builder = get_model(model_config['name'])
        expected_params = model_builder.__init__.__code__.co_varnames[1:]  # исключаем self
        filtered_config = {k: v for k, v in model_config.items() if k in expected_params}
        model = get_model(model_config['name'])(**filtered_config)


    except BaseException as e:
        full_traceback = traceback.format_exc()
        config_as_str = '\n'.join([f'{key}: {value}' for key, value in config.items()])
        telegram_send("Error preprocess: \n" + config_as_str +
                      '\n' + str(full_traceback))
        raise
    try:
        history = model.train(drop_nan_sub, drop_nan_sub)
    except BaseException as e:
        full_traceback = traceback.format_exc()
        config_as_str = '\n'.join([f'{key}: {value}' for key, value in config.items()])

        telegram_send("Error train: \n" + config_as_str +
                      '\n' + str(full_traceback))
        raise
    data_result = SerialImputeBehavior(model, window_size).simulate(data_normalize)
    result_mse = get_score(ScoreType.MSE)(dataset_blackout,
                                          data_result,
                                          dataset_origin_norm)

    np.savetxt(save_dir / 'result_impute.txt',
               normalizer.re_normalize(data_result))
    json.dump(result_mse.to_dict(), open(save_dir / 'score_result.json', 'w+'), indent=2)
    json.dump(history, open(save_dir / 'history.json', 'w+'), indent=2)
    json.dump(config, open(save_dir / 'config.json', 'w+'), indent=2)
    telegram_send("Start \n" + config_as_str + f"\n{result_mse}")


def load_config(filename: str) -> dict:
    """Загружает конфигурацию из файла JSON."""
    with open(filename, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start training with a given config file.")
    parser.add_argument('--config_file', type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    start_train(config)
