from dataloader import HPDatasetLoader
from metrics import Eval
from models import LLama3, Gemma, Phi3, Mistral
from prompts import Roleprompt, ZeroshotCoT, threeshotCoT, Leasttomost, generatedknowledge
from utils import AnswerProcessor
from abc import ABC
import argparse
from langchain import PromptTemplate, HuggingFacePipeline

#Prompts Dictionary
prompts = {
    1 : Roleprompt(),
    2 : ZeroshotCoT(),
    3 : threeshotCoT(),
    4 : Leasttomost(),
    5 : generatedknowledge()
}


class ManualHierarchicalPrompt(ABC):
    def __init__(self, model, dataset, metric, text_processor, prompts,task, prefix, suffix):
        self.model = model  
        self.dataset = dataset 
        self.metric = metric   
        self.text_processor = text_processor    
        self.prompts = prompts
        self.task = task   
        self.prefix = prefix  
        self.suffix = suffix 
    '''
    Prompt processing method
    '''
    def prompt_process(self,item):
        '''
        processes a single item from the dataset using hierarchical prompts
        '''
        level_Score = 1 # to track the score at each level
        for i in range(1,6):
            llm_f = self.model.generate_pipe_f()    # final pipeline
            llm_nf = self.model.generate_pipe_nf()  # non-final pipeline

            # handles passage and ques-ans pairs
            if self.task == "boolq":
                #extracting the passage, question, and answer from the item
                passage = item['passage']
                question = item['question']
                ans = item['answer']

                # level 4
                if i==4:
                    # retrieve multiple prompt strategies
                    templates = self.prompts[i].get_prompt(self.task)
                    predictions = ""
                    # iterate over the templates
                    for i in range(len(templates)):
                        # and create a prompt chain for each of them
                        prompt = PromptTemplate.from_template(templates[i])

                        # for intermediate templates, use llm_nf
                        if i != len(templates)-1:
                            chain = prompt | llm_nf
                            predictions = chain.invoke({'question': question,'passage':passage})
                        # for final template, use llm_f
                        else:
                            chain = prompt | llm_f
                            predictions = chain.invoke({'question': question,'passage':passage})
                elif i==5:
                    pass
                
                # for other levels, retrieve a single template, add the prefix and suffix, and create a prompt chain using llm_f
                else :
                    template = self.prompts[i].get_prompt(self.task)
                    template = self.prefix + template + self.suffix +"Answer:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    predictions = chain.invoke({'question': question,'passage':passage})

            # handles multiple-choice questions
            elif self.task == "csqa":
                # extract the question and choices 
                question = item['question'][0]
                text1 = item['choices'][0]['text'][0] 
                text2 = item['choices'][0]['text'][1] 
                text3 = item['choices'][0]['text'][2]
                text4 = item['choices'][0]['text'][3] 
                text5 = item['choices'][0]['text'][4] 

                # extract the answer key
                ans = item['answerKey']
                if i==4:
                    pass
                elif i==5:
                    pass
                else:
                    template = self.prompts[i].get_prompt(self.task)
                    template = self.prefix + template + self.suffix +"Answer:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    predictions = chain.invoke({'question': question,'text1':text1,'text2':text2,'text3':text3,'text4':text4,'text5':text5})

            # handles translation tasks
            elif self.task == "iwslt":
                # extract english text and answer in french
                eng_text = item['translation'][0]['en']
                answer  = item['translation']['fr']

                # retrieve a template and create a prompt
                template = self.prompts[i].get_prompt(self.task)
                prompt = PromptTemplate.from_template(template)

            # handles dialogue summarization tasks
            elif self.task == "samsum":
                # extract the dialogue and summary
                dialogue = item['dialogue']
                ans = item['summary']
                template = self.prompts[i].get_prompt(self.task)
                prompt = PromptTemplate.from_template(template)
            


class AdaptiveHierarchicalPrompt(ABC):
    def __init__(self, model, dataset, metric, text_processor, prompts, task, prefix, suffix):
        self.model = model
        self.dataset = dataset
        self.metric = metric
        self.text_processor = text_processor
        self.prompts = prompts
        self.task = task
        self.prefix = prefix
        self.suffix = suffix

    def select_prompt_level(self, item):
        # Use LLaMA3 or another model to choose the prompt level based on the item
        # Here, llm_selection_model is an instance of the model used for selecting the prompt level
        selection_prompt = f"Select the appropriate prompt level (1-5) for the following item: {item}"
        selected_level = self.model.select_prompt_level(selection_prompt)
        return int(selected_level.strip())

    def prompt_process(self, item):
        selected_level = self.select_prompt_level(item)
        llm_f = self.model.generate_pipe_f()
        llm_nf = self.model.generate_pipe_nf()

        passage = item['passage']
        question = item['question']
        ans = item['answer']
        
        if self.task == "boolq":
            if selected_level == 1:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(passage=passage, question=question)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'question': question, 'passage': passage})

            elif selected_level == 2:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(passage=passage, question=question)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'question': question, 'passage': passage})

            elif selected_level == 3:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(passage=passage, question=question)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'question': question, 'passage': passage})

            elif selected_level == 4:
                templates = self.prompts[selected_level].get_prompt(self.task)
                predictions = ""
                for i in range(len(templates)):
                    prompt_text = templates[i].format(passage=passage, question=question, predictions=predictions)
                    prompt = PromptTemplate.from_template(prompt_text)
                    if i != len(templates) - 1:
                        chain = prompt | llm_nf
                    else:
                        chain = prompt | llm_f
                    predictions = chain.invoke({'question': question, 'passage': passage})

            elif selected_level == 5:
                template, gen_knowledge_prompt = self.prompts[selected_level].get_prompt(self.task)
                knowledge_prompt_text = gen_knowledge_prompt.format(passage=passage)
                generated_knowledge = llm_nf.invoke({'passage': passage, 'question': question})
                prompt_text = template.format(passage=passage, question=question, pred=generated_knowledge)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'question': question, 'passage': passage})

        if self.task == 'csqa':
            question = item['question']
            choices = item['choices']
            text1 = choices[0]['text']
            text2 = choices[1]['text']
            text3 = choices[2]['text']
            text4 = choices[3]['text']
            text5 = choices[4]['text']
            ans = item['answerKey']

            if selected_level == 1:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

            elif selected_level == 2:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

            elif selected_level == 3:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

            elif selected_level == 4:
                templates = self.prompts[selected_level].get_prompt(self.task)
                predictions = ""
                for i in range(len(templates)):
                    prompt_text = templates[i].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, predictions=predictions)
                    prompt = PromptTemplate.from_template(prompt_text)
                    if i != len(templates) - 1:
                        chain = prompt | llm_nf
                    else:
                        chain = prompt | llm_f
                    predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

            elif selected_level == 5:
                template, gen_knowledge_prompt = self.prompts[selected_level].get_prompt(self.task)
                knowledge_prompt_text = gen_knowledge_prompt.format(question=question)
                generated_knowledge = llm_nf.invoke({'question': question})
                prompt_text = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred=generated_knowledge)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

        if self.task == 'iwslt':
            eng_text = item['translation']['en']
            fr_text = item['translation']['fr']

            if selected_level == 1:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(eng_text=eng_text)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'eng_text': eng_text})

            elif selected_level == 2:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(eng_text=eng_text)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'eng_text': eng_text})

            elif selected_level == 3:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(eng_text=eng_text)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'eng_text': eng_text})

            elif selected_level == 4:
                templates = self.prompts[selected_level].get_prompt(self.task)
                predictions = ""
                for i in range(len(templates)):
                    prompt_text = templates[i].format(eng_text=eng_text, predictions=predictions)
                    prompt = PromptTemplate.from_template(prompt_text)
                    if i != len(templates) - 1:
                        chain = prompt | llm_nf
                    else:
                        chain = prompt | llm_f
                    predictions = chain.invoke({'eng_text': eng_text})

            elif selected_level == 5:
                template, gen_knowledge_prompt = self.prompts[selected_level].get_prompt(self.task)
                knowledge_prompt_text = gen_knowledge_prompt.format(eng_text=eng_text)
                generated_knowledge = llm_nf.invoke({'eng_text': eng_text})
                prompt_text = template.format(eng_text=eng_text, pred=generated_knowledge)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'eng_text': eng_text})

        if self.task == 'samsum':
            dialogue = item['dialogue']
            summary = item['summary']

            if selected_level == 1:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(dialogue=dialogue)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'dialogue': dialogue})

            elif selected_level == 2:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(dialogue=dialogue)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'dialogue': dialogue})

            elif selected_level == 3:
                template = self.prompts[selected_level].get_prompt(self.task)
                prompt_text = template.format(dialogue=dialogue)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'dialogue': dialogue})

            elif selected_level == 4:
                templates = self.prompts[selected_level].get_prompt(self.task)
                predictions = ""
                for i in range(len(templates)):
                    prompt_text = templates[i].format(dialogue=dialogue, predictions=predictions)
                    prompt = PromptTemplate.from_template(prompt_text)
                    if i != len(templates) - 1:
                        chain = prompt | llm_nf
                    else:
                        chain = prompt | llm_f
                    predictions = chain.invoke({'dialogue': dialogue})

            elif selected_level == 5:
                template, gen_knowledge_prompt = self.prompts[selected_level].get_prompt(self.task)
                knowledge_prompt_text = gen_knowledge_prompt.format(dialogue=dialogue)
                generated_knowledge = llm_nf.invoke({'dialogue': dialogue})
                prompt_text = template.format(dialogue=dialogue, pred=generated_knowledge)
                prompt = PromptTemplate.from_template(prompt_text)
                chain = prompt | llm_f
                predictions = chain.invoke({'dialogue': dialogue})

        return predictions


def main(args):
    HP_framework = args.arg1
    model_name = args.arg2
    
    if model_name == "llama3":
        model = LLama3()
        prefix = "<|start_header_id|>user<|end_header_id|>\n"
        suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    elif model_name == "gemma":
        model = Gemma()
        prefix = "<bos><start_of_turn>user\n"
        suffix = "<end_of_turn>\n<start_of_turn>model\n"
    elif model_name == "phi3":
        model = Phi3()
        prefix = "<|endoftext|><|user|>\n"
        suffix = "<|end|>\n<|assistant|>\n"
    elif model_name == "mistral":
        model = Mistral()
        prefix = "<s>[INST]\n"
        suffix = "[/INST]\n"
    
    dataset_name = args.arg3
    if dataset_name in ["iwslt", "samsum"]:
        thres = args.thres

    data_loader = HPDatasetLoader()
    dataset = data_loader.get_dataset(dataset_name)
    text_processor = AnswerProcessor(dataset_name).processor
    eval = Eval(dataset_name).metric

    if HP_framework == "man":
        manual_hp = ManualHierarchicalPrompt(model, dataset, eval, text_processor, prompts, dataset_name, prefix, suffix)
    elif HP_framework == "auto":
        adaptive_hp = AdaptiveHierarchicalPrompt(model, dataset, eval, text_processor, prompts, dataset_name, prefix, suffix)
        for item in dataset:
            adaptive_hp.prompt_process(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for hierarchical prompt generation.')
    parser.add_argument('arg1', type=str, help='Manual or automatic prompt generation.')
    parser.add_argument('arg2', type=str, help='Model to be used.')
    parser.add_argument('arg3', type=str, help='Dataset to be used.')
    parser.add_argument('--thres', type=int, help='Threshold needed for iwslt and samsum datasets', default=0)
    args = parser.parse_args()
    main(args)