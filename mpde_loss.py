import argparse
import traceback

from AbstractModel.error.AbstractError import ErrorType, base_error_list
from base_loss import init_config
from start_experent import start_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start training with a given config file.")
    parser.add_argument('--model',
                        type=str,
                        help="Path to the JSON configuration file.")
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,  # Устанавливаем стандартное значение
                        help="Learning rate (default: 0.001)")
    parser.add_argument('--cuda',
                        type=int,
                        default=0,  # Устанавливаем стандартное значение
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
        config = init_config(dataset,
                             error_name="MPDE",
                             model_name=model_name)
        # config['epochs'] = 1
        config['lr'] = args.lr
        config['model']['device'] = f"cuda:{args.cuda}"
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
