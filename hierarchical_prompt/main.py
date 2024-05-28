from dataloader import DatasetLoader
from metrics import Eval
from models import LLama3, Gemma, Phi3, Mistral
from prompts import Roleprompt, ZeroshotCoT, threeshotCoT, Leasttomost, generatedknowledge
from utils import AnswerProcessor
from abc import ABC
import argparse
from langchain import PromptTemplate, HuggingFacePipeline

prompts = {
    1 : Roleprompt(),
    2 : ZeroshotCoT(),
    3 : threeshotCoT(),
    4 : Leasttomost(),
    5 : generatedknowledge()
}


class ManualHierarchicalPrompt(ABC):
    def __init__(self, model, dataset, metric, text_processor, prompts,task,interelation=False):
        self.model = model
        self.dataset = dataset
        self.metric = metric
        self.interelation = interelation
        self.text_processor = text_processor
        self.prompts = prompts
        self.task = task

    def prompt_process(self,item):
        level_Score = 1
        for i in range(1,6):
            llm_f = self.model.generate_pipe_f()
            llm_nf = self.model.generate_pipe_nf()
            if self.task == "boolq":
                passage = item['passage']
                question = item['question']
                ans = item['answer']
                if i==4:
                    templates = self.prompts[i].get_prompt(self.task)
                    predictions = ""
                    for i in range(len(templates)):
                        prompt = PromptTemplate.from_template(templates[i])

                        if i != len(templates)-1:
                            chain = prompt | llm_nf
                            predictions = chain.invoke({'question': question,'passage':passage})
                        else:
                            chain = prompt | llm_f
                            predictions = chain.invoke({'question': question,'passage':passage})

                template = self.prompts[i].get_prompt(self.task)
                prompt = PromptTemplate.from_template(template)
            elif self.task == "csqa":
                question = item['question'][0]
                text1 = item['choices'][0]['text'][0] 
                text2 = item['choices'][0]['text'][1] 
                text3 = item['choices'][0]['text'][2]
                text4 = item['choices'][0]['text'][3] 
                text5 = item['choices'][0]['text'][4] 
                ans = item['answerKey']
                template = self.prompts[i].get_prompt(self.task)
                prompt = PromptTemplate.from_template(template)

            elif self.task == "iwslt":
                eng_text = item['translation'][0]['en']
                answer  = item['translation']['fr']
                template = self.prompts[i].get_prompt(self.task)
                prompt = PromptTemplate.from_template(template)

            elif self.task == "samsum":
                dialogue = item['dialogue']
                ans = item['summary']
                template = self.prompts[i].get_prompt(self.task)
                prompt = PromptTemplate.from_template(template)
            

        
        

            


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
    if HP_framework == "man":
        manual_hp = ManualHierarchicalPrompt(model, dataset, eval, text_processor, prompts, interelation, dataset_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for hierarchical prompt generation.')
    parser.add_argument('arg1', type=str, help='Manual or automatic prompt generation.')
    parser.add_argument('arg2', type=str, help='Interelation between prompts.')
    parser.add_argument('arg3', type=str, help='model to be used.')
    parser.add_argument('arg4', type=str, help='dataset to be used.')
    parser.add_argument('--thres', type=int, help='threshold needed for some of the datasets', default=0)
    args = parser.parse_args()
    main(args)