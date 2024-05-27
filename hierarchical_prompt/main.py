from dataloader import DatasetLoader
from metrics import Eval
from models import LLama3, Gemma, Phi3, Mistral
from prompts import Roleprompt, ZeroshotCoT, threeshotCoT, Leasttomost, generatedknowledge
from utils import AnswerProcessor
from abc import ABC
import argparse
prompts = {
    "level1": Roleprompt(),
    "level2": ZeroshotCoT(),
    "level3": threeshotCoT(),
    "level4": Leasttomost(),
    "level5": generatedknowledge()
}


class ManualHierarchicalPrompt(ABC):
    def __init__(self, model, dataset, metric, answer_processor, prompts,interelation=False):
        self.model = model
        self.dataset = dataset
        self.metric = metric
        self.interelation = interelation
        self.answer_processor = answer_processor
        self.prompt = prompts


def main(args):
    HP_framework = args.arg1
    if(args.arg2 == "yes"):
        interelation = True
    model_name = args.arg3
    if model_name == "llama3":
        model = LLama3()
    elif model_name == "gemma":
        model = Gemma()
    elif model_name == "phi3":
        model = Phi3()
    elif model_name == "mistral":
        model = Mistral()
    dataset_name = args.arg4
    if(dataset_name == "iwslt" or dataset_name == "samsum"):
        thres = args.thres
    data_loader = DatasetLoader()
    dataset = data_loader.get_dataset(dataset_name)
    text_processor = AnswerProcessor(dataset_name).processor
    eval = Eval(dataset_name).metric



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for hierarchical prompt generation.')
    parser.add_argument('arg1', type=str, help='Manual or automatic prompt generation.')
    parser.add_argument('arg2', type=str, help='Interelation between prompts.')
    parser.add_argument('arg3', type=str, help='model to be used.')
    parser.add_argument('arg4', type=str, help='dataset to be used.')
    parser.add_argument('--thres', type=int, help='threshold needed for some of the datasets', default=0)
    args = parser.parse_args()
    main(args)