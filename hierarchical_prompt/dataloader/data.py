"""
 An utility class for loading the datasets for corresponding tasks.
"""

import logging
from datasets import load_dataset
from abc import ABC

logging.basicConfig(level=logging.INFO)

class DatasetLoader(ABC):
    def __init__(self):
        self.datasets = {
            "boolq": None,
            "csqa": None,
            "iwslt": None,
            "samsum": None
        }

    def load_datasets(self):
        # Load the datasets and store them in the dictionary
        self.datasets["boolq"] = load_dataset("google/boolq", split="validation")
        self.datasets["csqa"] = load_dataset("tau/commonsense_qa", split="validation")
        self.datasets["iwslt"] = load_dataset("iwslt2017", "iwslt2017-en-fr", split="validation")
        self.datasets["samsum"] = load_dataset("samsum", split="test")
    
    def get_dataset(self, name):
        if name not in self.datasets:
            logging.info(f'Dataset {name} not supported')
            return f'Dataset {name} not found'
        
        elif self.datasets[name] is None:
            logging.info(f' Dataset {name} is not loaded')
            try:
                self.load_datasets(name)
            except Exception as e:
                logging.error(f'Failed to load dataset {name}:{e}')
                return f'Failed to load dataset {name}:{e}'
        
        return self.datasets[name]
