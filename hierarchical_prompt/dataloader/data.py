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
        self.datasets["boolq"] = load_dataset("google/boolq", split="validation",trust_remote_code=True)
        self.datasets["csqa"] = load_dataset("tau/commonsense_qa", split="validation",trust_remote_code=True)
        self.datasets["iwslt"] = load_dataset("iwslt2017", "iwslt2017-en-fr", split="validation",trust_remote_code=True)
        self.datasets["samsum"] = load_dataset("samsum", split="test",trust_remote_code=True)
    
    def get_dataset(self, name):
        if name not in self.datasets:
            logging.info(f'Dataset {name} not supported')
            return f'Dataset {name} not found'
        
        self.load_datasets()
        logging.info(f"***Dataset {name} loaded successfully***")
        return self.datasets[name]
