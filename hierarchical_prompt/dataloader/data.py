"""
 An utility class for loading the datasets for corresponding tasks.
"""

import logging
from abc import ABC

from datasets import load_dataset


class DatasetLoader(ABC):
    def __init__(self):
        self.datasets = {
            "boolq": None,
            "csqa": None,
            "iwslt": None,
            "samsum": None,
            "humaneval": None,
            "gsm8k": None,
            "mmlu": None
        }

    def load_datasets(self, name):
        if name not in self.datasets:
            logging.info(f'Dataset {name} not supported')
            return f'Dataset {name} not found'

        if name == "boolq":
            self.datasets[name] = load_dataset("google/boolq", split="validation")
        elif name == "csqa":
            self.datasets[name] = load_dataset("tau/commonsense_qa", split="validation")
        elif name == "iwslt":
            self.datasets[name] = load_dataset("iwslt2017", "iwslt2017-en-fr", split="validation")
        elif name == "samsum":
            self.datasets[name] = load_dataset("samsum", split="test")
        elif name == "humaneval":
            self.datasets[name] = load_dataset("openai/openai_humaneval", split="test")
        elif name == "gsm8k":
            self.datasets[name] = load_dataset("openai/gsm8k", "main", split="test")
        elif name == "mmlu":
            self.datasets[name] = load_dataset("cais/mmlu", "all", split="test")
        else:
            logging.info(f'Dataset {name} not supported')
            return f'Dataset {name} not found'

        logging.info(f"***Dataset {name} loaded successfully***")
        return self.datasets[name]

    def get_dataset(self, name):
        if name not in self.datasets:
            logging.info(f'Dataset {name} not supported')
            return f'Dataset {name} not found'

        # Load dataset if not already loaded
        if self.datasets[name] is None:
            self.load_datasets(name)

        return self.datasets[name]
