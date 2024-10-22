import argparse
import json
import subprocess
import tempfile
from .start_model import start_train

from .start_array_json import load_config, init_config


def run_parallel_trainings(config_arr, max_parallel=4):
    processes = []

    for config in config_arr:
        if len(processes) >= max_parallel:
            finished_process = next(p for p in processes if p.poll() is not None)
            processes.remove(finished_process)

        p = subprocess.Popen(["python", "-m", "SANNI.start_config", json.dumps(config)])
        processes.append(p)

    # Ждем завершения всех оставшихся процессов
    for p in processes:
        p.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Start training with a given config file.")
    parser.add_argument('--config_file', type=str, help="Path to the JSON configuration file.")
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args()

    arr_config_path = args.config_file
    arr_config = load_config(arr_config_path)
    config_arr = []
    for config in arr_config:
        if config['error_name'] in ['MPDE', 'LogCosh']:
            continue
        print(config['model'])
        # if config['model'] in ['SANNI', 'SAETI']:
        #     continue
        config = init_config(f'{config["dataset"]}',
                             config["model"],
                             config["error_name"])
        config['model']['device'] = f'cuda:{args.cuda}'
        config_arr.append(config)
    # start_train(config_arr[0])

    run_parallel_trainings(config_arr)
