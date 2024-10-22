import json
import sys
from .start_model import start_train

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the config as a JSON string.")
        sys.exit(1)

    # Аргумент передан как JSON-строка
    config_str = sys.argv[1]

    try:
        # Парсим строку JSON в словарь
        config = json.loads(config_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

    # Запуск функции с полученной конфигурацией
    start_train(config)