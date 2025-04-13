import torch
from transformers import BitsAndBytesConfig
import json
from copy import deepcopy
from pprint import pformat
from enum import Enum

# Перечисления для допустимых значений
class ModelType(Enum):
    CAUSAL_LM = "causal_lm"
    SEQUENCE_CLASSIFICATION = "sequence_classification"

class TaskType(Enum):
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    NER = "ner"
    NONE = None

class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"

class BackgroundMode(Enum):
    THREAD = "thread"
    PROCESS = "process"
    SYNC = "sync"

class StreamerType(Enum):
    TEXT = "text"
    CUSTOM = "custom"

class LogLevel(Enum):
    INFO = "INFO"
    DEBUG = "DEBUG"
    ERROR = "ERROR"

class ConfigurationModel:
    configuration = {
        "model_characteristic": {
            "model_name": None,
            "model_type": ModelType.CAUSAL_LM.value,
            "task_type": TaskType.NONE.value
        },
        "model_options": {
            "device": Device.CPU.value,
            "quantization": {
                "have": False,
                "what": {
                    "4_bit": True,
                    "8_bit": False
                },
                "torch_dtype": {
                    "float16": True,
                    "float32": False
                }
            },
            "optimizations": {
                "compile_model": False
            }
        },
        "generation": {
            "max_new_tokens": 50,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True
        },
        "streaming": {
            "enable": False,
            "streamer_type": StreamerType.TEXT.value
        },
        "background": {
            "mode": BackgroundMode.THREAD.value
        },
        "logging": {
            "level": LogLevel.INFO.value,
            "file": None
        }
    }

    def __init__(self, config=None):
        """Инициализация с опциональной конфигурацией."""
        from copy import deepcopy

        self.configuration = deepcopy(self.configuration)

        if config:
            if isinstance(config, str):
                # Если передан путь к JSON-файлу
                self.load_from_json(config)
            elif isinstance(config, dict):
                # Если передан словарь
                self.update_configuration(**config)
            else:
                raise ValueError("Параметр 'config' должен быть либо словарем, либо путём к JSON-файлу.")



    @staticmethod
    def istrue(configuration):
        def _raise_config_error(location, message):
            """Метод для генерации ошибки конфигурации с указанием положения в словаре."""
            raise Exception(f"Недопустимая конфигурация: {location}: {message}")


        """Проверка конфигурации на корректность и взаимоисключения с использованием pattern matching."""
        # Проверка квантования
        match configuration["model_options"]["quantization"]:
            case {"have": False, "what": {"4_bit": four_bit, "8_bit": eight_bit}} if four_bit or eight_bit:
                _raise_config_error(
                    "model_options.quantization",
                    "'have' = False, но указаны параметры квантования:\n" +
                    str(configuration["model_options"]["quantization"]["what"])
                )
            case {"have": _, "what": {"4_bit": True, "8_bit": True}}:
                _raise_config_error(
                    "model_options.quantization.what",
                    "'4_bit' и '8_bit' не могут быть одновременно True:\n" +
                    str(configuration["model_options"]["quantization"]["what"])
                )
            case _:
                pass  # Другие случаи допустимы

        # Проверка torch_dtype
        match configuration["model_options"]["quantization"]["torch_dtype"]:
            case {"float16": True, "float32": True}:
               _raise_config_error(
                    "model_options.quantization.torch_dtype",
                    "'float16' и 'float32' не могут быть одновременно True:\n" +
                    str(configuration["model_options"]["quantization"]["torch_dtype"])
                )
            case _:
                pass  # Другие случаи допустимы

        # Проверка model_type
        model_type = configuration["model_characteristic"]["model_type"]
        match model_type in [e.value for e in ModelType]:
            case False:
                _raise_config_error(
                    "model_characteristic.model_type",
                    f"должно быть одним из {[e.value for e in ModelType]}, текущее: {model_type}"
                )
            case True:
                pass

        # Проверка task_type
        task_type = configuration["model_characteristic"]["task_type"]
        match task_type in [e.value for e in TaskType]:
            case False:
                _raise_config_error(
                    "model_characteristic.task_type",
                    f"должно быть одним из {[e.value for e in TaskType]}, текущее: {task_type}"
                )
            case True:
                pass

        # Проверка device
        device = configuration["model_options"]["device"]
        match device in [e.value for e in Device]:
            case False:
               _raise_config_error(
                    "model_options.device",
                    f"должно быть одним из {[e.value for e in Device]}, текущее: {device}"
                )
            case True:
                pass

        # Проверка background.mode
        background_mode = configuration["background"]["mode"]
        match background_mode in [e.value for e in BackgroundMode]:
            case False:
               _raise_config_error(
                    "background.mode",
                    f"должно быть одним из {[e.value for e in BackgroundMode]}, текущее: {background_mode}"
                )
            case True:
                pass

        # Проверка стриминга
        match configuration["streaming"]:
            case {"enable": True, "streamer_type": streamer_type} if streamer_type not in [e.value for e in StreamerType]:
                _raise_config_error(
                    "streaming.streamer_type",
                    f"должно быть одним из {[e.value for e in StreamerType]}, текущее: {streamer_type}"
                )
            case _:
                pass

        # Проверка логирования
        log_level = configuration["logging"]["level"]
        match log_level in [e.value for e in LogLevel]:
            case False:
               _raise_config_error(
                    "logging.level",
                    f"должно быть одним из {[e.value for e in LogLevel]}, текущее: {log_level}"
                )
            case True:
                pass

    def update_configuration(self, **settings):
        """Обновление конфигурации с проверкой корректности."""
        from copy import deepcopy

        try:
            # Создаём копию текущей конфигурации
            new_config = deepcopy(self.configuration)

            # Обновляем значения из переданных настроек
            def update_nested_dict(d, u):
                for key, value in u.items():
                    if isinstance(value, dict):
                        d[key] = update_nested_dict(d.get(key, {}), value)
                    else:
                        d[key] = value
                return d

            if settings:
                new_config = update_nested_dict(new_config, settings)

            # Проверяем корректность
            self.istrue(new_config)

            # Если всё ок, обновляем конфигурацию
            self.configuration = new_config
            print("Конфигурация успешно обновлена.")
            return self.configuration

        except Exception as e:
            print(f"Ошибка при обновлении конфигурации: {str(e)}")
            print("Выведите текущий шаблон конфигурации с помощью метода get_full_configuration(), чтобы корректно написать конфигурацию.")
            raise

    def load_from_json(self, json_path):

        """Загрузка конфигурации из JSON-файла."""


        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_config = json.load(f)

            # Проверяем, что структура JSON соответствует шаблону
            def check_structure(template, loaded):

                for key, value in template.items():
                    if key not in loaded:
                        raise Exception(f"Отсутствует ключ '{key}' в JSON-конфигурации.")
                    if isinstance(value, dict):
                        if not isinstance(loaded[key], dict):
                            raise Exception(f"Ключ '{key}' должен быть словарем, но получен тип {type(loaded[key])}.")
                        check_structure(value, loaded[key])
                    else:
                        if not isinstance(loaded[key], type(value)):
                            raise Exception(f"Ключ '{key}' должен быть типа {type(value)}, но получен тип {type(loaded[key])}.")

            check_structure(self.configuration, json_config)

            # Обновляем конфигурацию через update_nested_dict
            self.update_configuration(**json_config)

            print(f"Конфигурация успешно загружена из {json_path}.")
            return self.configuration

        except Exception as e:
            print(f"Ошибка при загрузке конфигурации из JSON: {str(e)}")
            raise

    def get_configuration(self):
        """Получение итоговой конфигурации без изменения настроек."""
        from transformers import BitsAndBytesConfig

        # Используем текущую конфигурацию
        config = self.configuration

        # Формируем итоговую конфигурацию
        result = {
            "model_characteristic": config["model_characteristic"],
            "model_options": {
                "device": config["model_options"]["device"],
                "quantization_config": None,
                "torch_dtype": torch.float16 if config["model_options"]["quantization"]["torch_dtype"]["float16"] else torch.float32,
                "compile_model": config["model_options"]["optimizations"]["compile_model"]
            },
            "generation": config["generation"],
            "streaming": config["streaming"],
            "background": config["background"],
            "logging": config["logging"]
        }

        # Если используется квантование, создаём BitsAndBytesConfig
        if config["model_options"]["quantization"]["have"]:
            quantization_type = "4bit" if config["model_options"]["quantization"]["what"]["4_bit"] else "8bit"
            torch_dtype = result["model_options"]["torch_dtype"]

            if quantization_type == "4bit":
                result["model_options"]["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype
                )
            else:
                result["model_options"]["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch_dtype
                )

        return result

    def get_full_configuration(self):
        """Возвращает полную текущую конфигурацию."""
        return self.configuration

    def __str__(self):
        """Строковое представление конфигурации."""
        from pprint import pformat
        return pformat(self.configuration, indent=4, width=20)