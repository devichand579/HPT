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
    def load_dataset_name(self, name, split):
        pass
    
    def get_dataset(self, name, split='validation'):
        if name not in self.datasets:
            logging.info(f'Dataset {name} not supported')
            return f'Dataset {name} not found'
        
        if self.datasets[name] is None:
            logging.info(f'Logging dataset {name} with split {split}')
            try:
                self.load_dataset_name(name, split)
            except Exception as e:
                logging.error(f'Failed to load dataset {name}:{e}')
                return f'Failed to load dataset {name}:{e}'
        
        return self.datasets[name]
    
class HPDatasetLoader(DatasetLoader):
    def load_dataset_name(self, name, split='validation'):
        if name == 'boolq':
            self.datasets[name] = load_dataset('google/boolq', split=split)
        elif name == 'csqa':
            self.datasets[name] = load_dataset('tau/commonsense_qa', split=split)
        elif name == 'iwslt':
            self.datasets[name] = load_dataset('iwslt2017','iwslt2017-en-fr', split=split)
        elif name == 'samsum':
            self.datasets[name] = load_dataset('samsum', split=split)
        else:
            return ValueError(f'Dataset {name} not supported')
        
# Example usage
# dataset_loader = HPDatasetLoader()
# boolq_dataset = dataset_loader.get_dataset("boolq")
# csqa_dataset = dataset_loader.get_dataset("csqa")
