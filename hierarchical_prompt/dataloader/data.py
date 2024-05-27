"""
 An utility class for loading the datasets for corresponding tasks.
"""


from datasets import load_dataset
from abc import ABC

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
        if self.datasets[name] is None:
            self.load_datasets()
        return self.datasets.get(name, f"Dataset '{name}' not found")