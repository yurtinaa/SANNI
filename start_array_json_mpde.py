import argparse
import json
import torch.multiprocessing as mp
from .start_model import start_train
from enum import Enum

base_config = {
    "batch_size": 256,
    "epochs": 100,
    "lr": 0.0005,
    "dataset": "",
    "windows": 100,
    "error": {},
    "model": {},
}

error_config = {
    "type": "MSE",
    "params": {
    }
}

base_model_config = {
    "name": f"",
    # "device": "cuda:0",
}

sanni_model_config = {
    'name': 'SANNI',
    "snippet_count": 2
}

specific_model_config = {
    "SANNI": sanni_model_config,
    "SAETI": sanni_model_config
}


def init_model(model_name):
    model_config = specific_model_config.get(model_name,
                                             base_model_config).copy()
    model_config['name'] = model_name
    return model_config


def init_error(error_name):
    error = error_config.copy()
    error['type'] = error_name
    if error_name == 'MPDE':
        error['params'] = {"windows": 50,
                           "mse": True,
                           "alpha_beta": [0.25, 0.75]}
    return error


def init_config(dataset_name: str, model_name: str, error_name):
    config = base_config.copy()
    config["dataset"] = f"datasets/{dataset_name}"
    config_model = init_model(model_name)
    config['model'] = config_model
    config['error'] = init_error(error_name)
    return config


def load_config(filename: str) -> dict:
    """Загружает конфигурацию из файла JSON."""
    with open(filename, 'r') as f:
        return json.load(f)


def train(config):
    print(config)


def worker(queue):
    while not queue.empty():
        config = queue.get()
        print('start', config)
        start_train(config)


def run_parallel_trainings(config_arr, max_parallel=4):
    mp.set_start_method('spawn', force=True)  # Устанавливаем метод инициализации процессов

    # Создаем очередь для конфигураций
    queue = mp.Queue()

    # Заполняем очередь конфигурациями
    for config in config_arr:
        queue.put(config)

    # Создаем и запускаем процессы
    processes = []
    for _ in range(min(max_parallel, len(config_arr))):
        p = mp.Process(target=worker, args=(queue,))
        p.start()
        processes.append(p)

    # Дожидаемся завершения всех процессов
    for p in processes:
        p.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Start training with a given config file.")
    parser.add_argument('--config_file', type=str, help="Path to the JSON configuration file.")
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--parall', type=int, default=4)

    args = parser.parse_args()

    arr_config_path = args.config_file
    arr_config = load_config(arr_config_path)
    config_arr = []
    for config in arr_config:
        # print(config['model'])
        # if config['model'] in ['SANNI', 'SAETI']:
        #     continue
        alpha_beta = [config['alpha'], config['beta']]
        config = init_config(f'{config["dataset"]}',
                             config["model"],
                             'MPDE')
        config['model']['device'] = f'cuda:{args.cuda}'
        config['error']['params']["alpha_beta"] = alpha_beta
        config_arr.append(config)
    # start_train(config_arr[0])
    # print(config_arr[0])
    run_parallel_trainings(config_arr,max_parallel=args.parall)
