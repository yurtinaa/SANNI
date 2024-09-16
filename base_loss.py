import argparse
import traceback

from AbstractModel.error.AbstractError import ErrorType, base_error_list
from start_experent import start_train

base_config = {
    "batch_size": 256,
    "epochs": 500,
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
    "device": "cuda:0",
}

sanni_model_config = {
    'name': 'SANNI',
    "device": "cuda:0",
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
    return error


def init_config(dataset_name: str, model_name: str, error_name):
    config = base_config.copy()
    config["dataset"] = f"datasets/{dataset_name}"
    config_model = init_model(model_name)
    config['model'] = config_model
    config['error'] = init_error(error_name)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start training with a given config file.")
    parser.add_argument('--model',
                        type=str,
                        help="Path to the JSON configuration file.")
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,  # Устанавливаем стандартное значение
                        help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    dataset_list = ['airq',
                    'bafu',
                    'climate',
                    'electricity',
                    'germany',
                    'madrid',
                    'meteo',
                    'nrel',
                    'pamap',
                    'walk_run']
    error_list = base_error_list
    model_name = args.model
    model_name: str
    model_name = model_name.upper()
    for dataset in dataset_list:
        for error_name in error_list:
            config = init_config(dataset,
                                 error_name=error_name,
                                 model_name=model_name)
            config['lr'] = args.lr
            try:
                start_train(config)
            except Exception as e:
                print("An error occurred:")
                print(f"Error: {str(e)}")

                # Выводим полный стектрейс ошибки
                full_traceback = traceback.format_exc()
                print("Traceback details:")
                print(full_traceback)
                raise