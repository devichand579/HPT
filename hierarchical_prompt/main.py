from dataloader import DatasetLoader
from metrics import Eval
from models import LLama3, Gemma, Phi3, Mistral, Nemo, Gemma2, GPT4o, Claude
from prompts import Roleprompt, ZeroshotCoT, threeshotCoT, Leasttomost, generatedknowledge, Promptloader
from utils import AnswerProcessor, AdaptiveProcessor
from abc import ABC
import argparse
import json
import logging
import re

prompts = {
    1 : Roleprompt(),
    2 : ZeroshotCoT(),
    3 : threeshotCoT(),
    4 : Leasttomost(),
    5 : generatedknowledge()
}

hp_scores = {
    "boolq": 1.71,
    "csqa": 2.52,
    "iwslt": 1.92,
    "samsum": 2.23,
    "humaneval":4.68,
    "gsm8k": 2.14,
    "mmlu": 3.03
}




class ManualHierarchicalPrompt(ABC):
    def __init__(self, model, gen_model, dataset, metric_list, text_processor, prompts,task, prefix, suffix,thres=0):
        self.model = model 
        self.gen_model = gen_model 
        self.dataset = dataset 
        self.metrics = metric_list  
        self.text_processor = text_processor    
        self.prompts = prompts
        self.task = task   
        self.prefix = prefix  
        self.suffix = suffix 
        self.thres = thres
        self.predictions = []
        self.references = []
        self.scores = []
    '''
    Prompt processing method
    '''
    def prompt_process(self,item):
        '''
        processes a single item from the dataset using hierarchical prompts
        '''
        level = 1
        for i in range(1,6):
    
            llm_f = self.model.pipe_f   # full_text pipeline
            llm_nf = self.model.pipe_nf # non_full_text pipeline

            if self.task == "mmlu":
                # extract the question and choices 
                question = item['question']
                text1 = item['choices']['text'][0] 
                text2 = item['choices']['text'][1] 
                text3 = item['choices']['text'][2]
                text4 = item['choices']['text'][3] 
                # extract the answer 
                answer = item['answer']
                if answer == "A":
                    ans = 0
                elif answer == "B":
                    ans = 1
                elif answer == "C":
                    ans = 2
                elif answer == "D":
                    ans = 3
                # level 4
                if i==4:
                    # retrieve multiple levels of least-to-most prompting
                    templates = self.prompts[i].get_prompt(self.task)
                    pred_text = ""
                    # iterate over the templates
                    for i in range(len(templates)):
                        
                        template = self.prefix + templates[i].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, pred=pred_text) + self.suffix + "Answer:"

                        # for intermediate templates, use llm_nf
                        if i != len(templates)-1:
                            pred = llm_nf(template)
                            pred_text = pred[0]['generated_text']
                        # for final template, use llm_f
                        else:
                            pred = llm_f(template)
                            pred_text = pred[0]['generated_text']
                    # process the prediction
                    final_ans = self.text_processor(pred_text)
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + 1
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(question=question)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    
                    template = self.prefix + template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred = generated_knowledge) + self.suffix + "Answer:"
                    pred = llm_f(template)

                    # process the prediction
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + hp_scores[self.task]
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                    
                
                else :
                    template = self.prompts[i].get_prompt(self.task).format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
                    template = self.prefix + template + self.suffix +"Answer:"
                    pred = llm_f(template)
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + 1
                        continue


            if self.task == "humaneval":
                code = item['code']
                test_case = item['test']

                if i==4:
                    pred_txt = ""
                    templates = self.prompts[i].get_prompt(self.task)
                    for i in range(len(templates)):
                        template = self.prefix + templates[i].format(code=code, pred=pred_txt) + self.suffix + "Code:"
                        if i != len(templates)-1:
                            pred = llm_nf(template)
                            pred_txt = pred[0]['generated_text']
                        else:
                            pred = llm_f(template)
                            pred_txt = pred[0]['generated_text']

                    #process the prediction
                    final_ans = self.text_processor(pred_txt)
                    code_eval = self.metrics[0]
                    eval_score = code_eval([final_ans],[test_case])
                    if eval_score == 1:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(test_case)
                        break
                    else:
                        level = level + 1
                        continue

                elif i == 5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(passage=passage)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    
                    template = self.prefix + template.format(passage=passage, question=question, pred = generated_knowledge) + self.suffix + "Code:"
                    pred = llm_f(template)
                    # process the prediction
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    code_eval = self.metrics[0]
                    eval_score = code_eval([final_ans],[test_case])
                    if eval_score == 1:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(test_case)
                        break
                    else:
                        level = level + hp_scores[self.task]
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(test_case)
                     
                else:
                    template = self.prompts[i].get_prompt(self.task).format(passage=passage, question=question)
                    template = self.prefix + template + self.suffix + "Code:"
                    pred = llm_f(template)
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    code_eval = self.metrics[0]
                    eval_score = code_eval([final_ans],[test_case])
                    if eval_score == 1:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(test_case)
                        break
                    else:
                        level = level + 1
                        continue

            if self.task == "gsm8k":
                question = item['question']
                answer = item['answer'].split('#### ')[-1].strip()

                if i==4:
                    pred_txt = ""
                    templates = self.prompts[i].get_prompt(self.task)
                    for i in range(len(templates)):
                        template = self.prefix + templates[i].format(question=question, pred=pred_txt) + self.suffix + "Answer:"
                        if i != len(templates)-1:
                            pred = llm_nf(template)
                            pred_txt = pred[0]['generated_text']
                        else:
                            pred = llm_f(template)
                            pred_txt = pred[0]['generated_text']
                    final_ans = self.text_processor(pred_txt)
                    if final_ans == answer:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + 1
                        continue
                
                elif i == 5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(question=question)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
                    

                    template = self.prefix + template.format(question=question, pred = generated_knowledge) + self.suffix + "Answer:"
                    pred = llm_f(template)
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    if final_ans == answer:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + hp_scores[self.task]
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                
                else:
                    template = self.prompts[i].get_prompt(self.task).format(question=question)
                    template = self.prefix + template + self.suffix + "Answer:"
                    pred = llm_f(template)
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    if final_ans == answer:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + 1
                        continue
            # handles passage and ques-ans pairs
            if self.task == "boolq":
                #extracting the passage, question, and answer from the item
                passage = item['passage']
                question = item['question']
                answer = item['answer']
                if answer == True:
                    ans = 1
                else:
                    ans = 0

                # level 4
                if i==4:
                    # retrieve multiple levels of least-to-most prompting
                    pred_txt = ""
                    templates = self.prompts[i].get_prompt(self.task)
                    # iterate over the templates
                    for i in range(len(templates)):
                        
                        template = self.prefix + templates[i].format(passage=passage, question=question, pred=pred_txt) + self.suffix + "Answer:"
                        # for intermediate templates, use llm_nf
                        if i != len(templates)-1:
                            pred = llm_nf(template)
                            pred_txt = pred[0]['generated_text']
                        # for final template, use llm_f
                        else:
                            pred = llm_f(template)
                            pred_txt = pred[0]['generated_text']
                    # process the prediction
                    final_ans = self.text_processor(pred_txt)
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + 1
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    
                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(passage=passage)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    
                    template = self.prefix + template.format(passage=passage, question=question, pred = generated_knowledge) + self.suffix + "Answer:"
                    pred = llm_f(template)
                    # process the prediction
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + hp_scores[self.task]
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                    
                
                else :
                    template = self.prompts[i].get_prompt(self.task).format(passage=passage, question=question)
                    template = self.prefix + template + self.suffix + "Answer:"
                    pred = llm_f(template)
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + 1
                        continue

            # handles multiple-choice questions
            elif self.task == "csqa":
                # extract the question and choices 
                question = item['question']
                text1 = item['choices']['text'][0] 
                text2 = item['choices']['text'][1] 
                text3 = item['choices']['text'][2]
                text4 = item['choices']['text'][3] 
                text5 = item['choices']['text'][4]  
                # extract the answer key
                answer = item['answerKey']
                if answer == "A":
                    ans = 0
                elif answer == "B":
                    ans = 1
                elif answer == "C":
                    ans = 2
                elif answer == "D":
                    ans = 3
                elif answer == "E":
                    ans = 4
                # level 4
                if i==4:
                    # retrieve multiple levels of least-to-most prompting
                    templates = self.prompts[i].get_prompt(self.task)
                    pred_text = ""
                    # iterate over the templates
                    for i in range(len(templates)):
                        
                        template = self.prefix + templates[i].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred=pred_text) + self.suffix + "Answer:"

                        # for intermediate templates, use llm_nf
                        if i != len(templates)-1:
                            pred = llm_nf(template)
                            pred_text = pred[0]['generated_text']
                        # for final template, use llm_f
                        else:
                            pred = llm_f(template)
                            pred_text = pred[0]['generated_text']
                    # process the prediction
                    final_ans = self.text_processor(pred_text)
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + 1
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(question=question)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    
                    template = self.prefix + template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred = generated_knowledge) + self.suffix + "Answer:"
                    pred = llm_f(template)

                    # process the prediction
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + hp_scores[self.task]
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                    
                
                else :
                    template = self.prompts[i].get_prompt(self.task).format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
                    template = self.prefix + template + self.suffix +"Answer:"
                    pred = llm_f(template)
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    if final_ans == ans:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        level = level + 1
                        continue

            # handles translation tasks
            elif self.task == "iwslt":
                # extract english text and answer in french
                eng_text = item['translation']['en']
                answer  = item['translation']['fr']
                # level 4
                if i==4:
                    # retrieve multiple levels of least-to-most prompting
                    templates = self.prompts[i].get_prompt(self.task)
                    pred_text = ""
                    # iterate over the templates
                    for i in range(len(templates)):
                        
                        if i != len(templates)-1:
                            template = self.prefix + templates[i].format(eng_text=eng_text, pred=pred_text) + self.suffix
                            pred = llm_nf(template)
                            pred_text = pred[0]['generated_text']
                        # for final template, use llm_f
                        else:
                            template = self.prefix + templates[i].format(eng_text=eng_text, pred=pred_text) + self.suffix + "French:"
                            pred = llm_f(template)
                            pred_text = pred[0]['generated_text']
                    # process the prediction
                    final_ans = self.text_processor(pred_text)
                    bleu_score  = self.metrics[0]
                    eval_score = bleu_score([final_ans],[answer])
                    if  eval_score >= self.thres:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + 1
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(eng_text=eng_text)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    # create the final prompt and chain using llm_f
                    template = self.prefix + template.format(eng_text=eng_text, pred = generated_knowledge) + self.suffix + "French:"
                    pred = llm_f(template)
                    # process the prediction
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    bleu_score  = self.metrics[0]
                    eval_score = bleu_score([final_ans],[answer])
                    if  eval_score >= self.thres:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + hp_scores[self.task]
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                
                
                # for other levels, retrieve the prompt template, add the prefix and suffix, and create a prompt chain using llm_f
                else :
                    template = self.prompts[i].get_prompt(self.task).format(eng_text=eng_text)
                    template = self.prefix + template + self.suffix + "French:"
                    pred = llm_f(template)
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    bleu_score  = self.metrics[0]
                    eval_score = bleu_score([final_ans],[answer])
                    if  eval_score >= self.thres:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + 1
                        continue


            # handles dialogue summarization tasks
            elif self.task == "samsum":
                # extract the dialogue and summary
                dialogue = item['dialogue']
                answer = item['summary']
                
                # level 4
                if i==4:
                    # retrieve multiple levels of least-to-most prompting
                    templates = self.prompts[i].get_prompt(self.task)
                    pred_text = ""
                    # iterate over the templates
                    for i in range(len(templates)):
                        
                        if i != len(templates)-1:
                            template = self.prefix + templates[i].format(dialogue=dialogue, pred=pred_text) + self.suffix
                            pred = llm_nf(template)
                            pred_text = pred[0]['generated_text']
                        # for final template, use llm_f
                        else:
                            template = self.prefix + templates[i].format(dialogue=dialogue, pred=pred_text) + self.suffix + "Summary:"
                            pred = llm_f(template)
                            pred_text = pred[0]['generated_text']
                    # process the prediction
                    final_ans = self.text_processor(pred_text)
                    rogue_score  = self.metrics[0]
                    eval_score = rogue_score([final_ans],[answer])
                    if  eval_score["rouge1"] >= self.thres:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + 1
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(dialogue=dialogue)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                  
                    template = self.prefix + template.format(dialogue=dialogue, pred = generated_knowledge) + self.suffix + "Summary:"
                    pred = llm_f(template)
                    # process the prediction
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    rogue_score  = self.metrics[0]
                    eval_score = rogue_score([final_ans],[answer])
                    if  eval_score["rouge1"] >= self.thres:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + hp_scores[self.task]
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                
                
                # for other levels, retrieve the prompt template, add the prefix and suffix, and create a prompt chain using llm_f
                else :
                    template = self.prompts[i].get_prompt(self.task).format(dialogue=dialogue)
                    template = self.prefix + template + self.suffix + "Summary:"
                    pred = llm_f(template)
                    final_ans = self.text_processor(pred[0]['generated_text'])
                    rogue_score  = self.metrics[0]
                    eval_score = rogue_score([final_ans],[answer])
                    if  eval_score["rouge1"] >= self.thres:
                        self.scores.append(level)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        level = level + 1
                        continue
    def process_dataset(self):
        '''
        processes the entire dataset using hierarchical prompts
        '''
        for item in self.dataset:
            self.prompt_process(item)
        logging.info("***Dataset processed successfully***")

    
    def compute_scores(self):
        '''
        computes the scores for the predictions
        '''
        hp_score  = sum(self.scores)/len(self.scores)
        if self.task == "boolq":
            acc = self.metrics[0](self.predictions,self.references)
            f1 = self.metrics[1](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "accuracy": acc,
                "f1": f1
            }
            return scores
        elif self.task == "csqa":
            acc = self.metrics[0](self.predictions,self.references)
            f1 = self.metrics[1](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "accuracy": acc,
                "f1": f1
            }
            return scores
        elif self.task == "iwslt":
            bleu = self.metrics[0](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "bleu": bleu
            }
            return scores
        elif self.task == "samsum":
            rouge = self.metrics[0](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "rouge": rouge
            }
            return scores
        elif self.task == "gsm8k":
            acc = self.metrics[0](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "accuracy": acc
            }
            return scores
        elif self.task == "humaneval":
            pass_at_k = self.metrics[0](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "pass_at_k": pass_at_k
            }
            return scores
            


class AdaptiveHierarchicalPrompt(ABC):
    def __init__(self, model, gen_model, dataset, metric_list, text_processor, prompts,task, prefix, suffix,thres=0):
        self.model = model 
        self.gen_model = gen_model 
        self.dataset = dataset 
        self.metrics = metric_list  
        self.text_processor = text_processor    
        self.prompts = prompts
        self.task = task   
        self.prefix = prefix  
        self.suffix = suffix 
        self.thres = thres
        self.predictions = []
        self.references = []
        self.scores = []
        self.basic_tasks = {
                "boolq":  ("Based on the passage:'{0}'\nAnswer True/False to the question: '{1}'").format("{passage}", "{question}"),
                "csqa":   ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4}\nE {5}.").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{text5}"),
                "iwslt":  ("Translate '{0}' to french.").format("{eng_text}"),
                "samsum": ("Summarise the Dialogue: '{0}'.").format("{dialogue}") 
        }
        self.prompt_loader = Promptloader()
        self.adaptive_processor = AdaptiveProcessor().processor

    def select_prompt_level(self,item,prev_res = ""):
        '''
        selects the prompt level based on the item
        '''
        llm_f = self.gen_model.pipe_f   # full_text pipeline
        gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
        gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        if self.task == "boolq":
            passage = item['passage']
            question = item['question']
            template  = self.prompt_loader.adaptive_prompt
            task_template = self.basic_tasks[self.task].format(passage=passage, question=question)
            template = gen_prefix + template.format(prev_res=prev_res ,task = task_template) + gen_suffix + "Level:"
            pred = llm_f(template)
        if self.task == "csqa":
            question = item['question']
            text1 = item['choices']['text'][0] 
            text2 = item['choices']['text'][1] 
            text3 = item['choices']['text'][2]
            text4 = item['choices']['text'][3] 
            text5 = item['choices']['text'][4]  
            template  = self.prompt_loader.adaptive_prompt
            task_template = self.basic_tasks[self.task].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
            template = gen_prefix + template.format(prev_res=prev_res ,task = task_template) + gen_suffix + "Level:"
            pred = llm_f(template)
        if self.task == "iwslt":
            eng_text = item['translation']['en']
            template  = self.prompt_loader.adaptive_prompt
            task_template = self.basic_tasks[self.task].format(eng_text=eng_text)
            template = gen_prefix + template.format(prev_res=prev_res ,task = task_template) + gen_suffix + "Level:"
            pred = llm_f(template)
        if self.task == "samsum":
            dialogue = item['dialogue']
            template  = self.prompt_loader.adaptive_prompt
            task_template = self.basic_tasks[self.task].format(dialogue=dialogue)
            template = gen_prefix + template.format(prev_res=prev_res ,task = task_template) + gen_suffix + "Level:"
            pred = llm_f(template)
        level_txt = self.adaptive_processor(pred[0]['generated_text'])
        match = re.search(r'\d+', level_txt)
        if match:
            level  = int(match.group())
        else :
            level = 1
        return level

    '''
    Prompt processing method
    '''
    def prompt_process(self,item,level):
        '''
        processes a single item from the dataset using hierarchical prompts
        '''
        llm_f = self.model.pipe_f   # full_text pipeline
        llm_nf = self.model.pipe_nf # non_full_text pipeline
        final_ans = ""
        if level ==1 or level == 2 or level == 3:
            # handles passage and ques-ans pairs
            i=level
            if self.task == "boolq":
                #extracting the passage, question, and answer from the item
                passage = item['passage']
                question = item['question']
                answer = item['answer']
                if answer == True:
                    ans = 1
                else:
                    ans = 0
                template = self.prompts[i].get_prompt(self.task).format(passage=passage, question=question)
                template = self.prefix + template + self.suffix + "Answer:"
                pred = llm_f(template)
                final_ans = self.text_processor(pred[0]['generated_text'])
                if final_ans == ans:
                    self.predictions.append(final_ans)
                    self.references.append(ans)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
            if self.task == "csqa":
               # extract the question and choices 
                question = item['question']
                text1 = item['choices']['text'][0] 
                text2 = item['choices']['text'][1] 
                text3 = item['choices']['text'][2]
                text4 = item['choices']['text'][3] 
                text5 = item['choices']['text'][4]  
                # extract the answer key
                answer = item['answerKey']
                if answer == "A":
                    ans = 0
                elif answer == "B":
                    ans = 1
                elif answer == "C":
                    ans = 2
                elif answer == "D":
                    ans = 3
                elif answer == "E":
                    ans = 4
                template = self.prompts[i].get_prompt(self.task).format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
                template = self.prefix + template + self.suffix +"Answer:"
                pred = llm_f(template)
                final_ans = self.text_processor(pred[0]['generated_text'])
                if final_ans == ans:
                    self.predictions.append(final_ans)
                    self.references.append(ans)
                    return level, pred[0]['generated_text'] 
                else:
                    return 0, pred[0]['generated_text'] 
            if self.task == "iwslt":
                # extract english text and answer in french
                eng_text = item['translation']['en']
                answer  = item['translation']['fr']
                template = self.prompts[i].get_prompt(self.task).format(eng_text=eng_text)
                template = self.prefix + template + self.suffix + "French:"
                pred = llm_f(template)
                final_ans = self.text_processor(pred[0]['generated_text'])
                bleu_score  = self.metrics[0]
                eval_score = bleu_score([final_ans],[answer])
                if  eval_score >= self.thres:
                    self.predictions.append(final_ans)
                    self.references.append(answer)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
            if self.task == "samsum":
                # extract the dialogue and summary
                dialogue = item['dialogue']
                answer = item['summary']
                template = self.prompts[i].get_prompt(self.task).format(dialogue=dialogue)
                template = self.prefix + template + self.suffix + "Summary:"
                pred = llm_f(template)
                final_ans = self.text_processor(pred[0]['generated_text'])
                rogue_score  = self.metrics[0]
                eval_score = rogue_score([final_ans],[answer])
                if  eval_score["rouge1"] >= self.thres:
                    self.predictions.append(final_ans)
                    self.references.append(answer)
                    return level,pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
                

        if level==4:
            # handles passage and ques-ans pairs
            i = level
            if self.task == "boolq":
                #extracting the passage, question, and answer from the item
                passage = item['passage']
                question = item['question']
                answer = item['answer']
                if answer == True:
                    ans = 1
                else:
                    ans = 0
                # retrieve multiple levels of least-to-most prompting
                pred_txt = ""
                templates = self.prompts[i].get_prompt(self.task)
                # iterate over the templates
                for i in range(len(templates)):
                    template = self.prefix + templates[i].format(passage=passage, question=question, pred=pred_txt) + self.suffix + "Answer:"
                    # for intermediate templates, use llm_nf
                    if i != len(templates)-1:
                        pred = llm_nf(template)
                        pred_txt = pred[0]['generated_text']
                    # for final template, use llm_f
                    else:
                        pred = llm_f(template)
                        pred_txt = pred[0]['generated_text']
                # process the prediction
                final_ans = self.text_processor(pred_txt)
                if final_ans == ans:
                    self.predictions.append(final_ans)
                    self.references.append(ans)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
            if self.task == "csqa":
                # extract the question and choices
                question = item['question']
                text1 = item['choices']['text'][0]
                text2 = item['choices']['text'][1]
                text3 = item['choices']['text'][2]
                text4 = item['choices']['text'][3]
                text5 = item['choices']['text'][4]
                # extract the answer key
                answer = item['answerKey']
                if answer == "A":
                    ans = 0
                elif answer == "B":
                    ans = 1
                elif answer == "C":
                    ans = 2
                elif answer == "D":
                    ans = 3
                elif answer == "E":
                    ans = 4
                # retrieve multiple levels of least-to-most prompting
                templates = self.prompts[i].get_prompt(self.task)
                pred_text = ""  
                # iterate over the templates
                for i in range(len(templates)):
                    template = self.prefix + templates[i].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred=pred_text) + self.suffix + "Answer:"
                    # for intermediate templates, use llm_nf
                    if i != len(templates)-1:
                        pred = llm_nf(template)
                        pred_text = pred[0]['generated_text']
                    # for final template, use llm_f
                    else:
                        pred = llm_f(template)
                        pred_text = pred[0]['generated_text']
                # process the prediction
                final_ans = self.text_processor(pred_text)
                if final_ans == ans:
                    self.predictions.append(final_ans)
                    self.references.append(ans)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
            if self.task == "iwslt":
                # extract english text and answer in french
                eng_text = item['translation']['en']
                answer = item['translation']['fr']
                # retrieve multiple levels of least-to-most prompting
                templates = self.prompts[i].get_prompt(self.task)
                pred_text = ""
                # iterate over the templates
                for i in range(len(templates)):
                    if i != len(templates)-1:
                        template = self.prefix + templates[i].format(eng_text=eng_text, pred=pred_text) + self.suffix
                        pred = llm_nf(template)
                        pred_text = pred[0]['generated_text']
                    # for final template, use llm_f
                    else:
                        template = self.prefix + templates[i].format(eng_text=eng_text, pred=pred_text) + self.suffix + "French:"
                        pred = llm_f(template)
                        pred_text = pred[0]['generated_text']
                # process the prediction
                final_ans = self.text_processor(pred_text)
                bleu_score = self.metrics[0]
                eval_score = bleu_score([final_ans], [answer])
                if eval_score >= self.thres:
                    self.predictions.append(final_ans)
                    self.references.append(answer)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
            if self.task == "samsum":
                # extract the dialogue and summary
                dialogue = item['dialogue']
                answer = item['summary']
                # retrieve multiple levels of least-to-most prompting
                templates = self.prompts[i].get_prompt(self.task)
                pred_text = ""
                # iterate over the templates
                for i in range(len(templates)):
                    if i != len(templates)-1:
                        template = self.prefix + templates[i].format(dialogue=dialogue, pred=pred_text) + self.suffix
                        pred = llm_nf(template)
                        pred_text = pred[0]['generated_text']
                    # for final template, use llm_f
                    else:
                        template = self.prefix + templates[i].format(dialogue=dialogue, pred=pred_text) + self.suffix + "Summary:"
                        pred = llm_f(template)
                        pred_text = pred[0]['generated_text']
                # process the prediction
                final_ans = self.text_processor(pred_text)
                rogue_score = self.metrics[0]
                eval_score = rogue_score([final_ans], [answer])
                if eval_score["rouge1"] >= self.thres:
                    self.predictions.append(final_ans)
                    self.references.append(answer)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
        if level == 5:
            # handles passage and ques-ans pairs
            i = level
            if self.task == "boolq":
                #extracting the passage, question, and answer from the item
                passage = item['passage']
                question = item['question']
                answer = item['answer']
                if answer == True:
                    ans = 1
                else:
                    ans = 0
                gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    
                template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                gen_knowledge_template = gen_knowledge_template.format(passage=passage)
                knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                know_prompts_list = []
                for i in range(3):
                    know_prompts_list.append(knowledge_template)
                generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    
                template = self.prefix + template.format(passage=passage, question=question, pred = generated_knowledge) + self.suffix + "Answer:"
                pred = llm_f(template)
                # process the prediction
                final_ans = self.text_processor(pred[0]['generated_text'])   
                if final_ans == ans:
                    self.predictions.append(final_ans)
                    self.references.append(ans)
                    return level,pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
            if self.task == "csqa":
                # extract the question and choices 
                question = item['question']
                text1 = item['choices']['text'][0] 
                text2 = item['choices']['text'][1] 
                text3 = item['choices']['text'][2]
                text4 = item['choices']['text'][3] 
                text5 = item['choices']['text'][4]  
                # extract the answer key
                answer = item['answerKey']
                if answer == "A":
                    ans = 0
                elif answer == "B":
                    ans = 1
                elif answer == "C":
                    ans = 2
                elif answer == "D":
                    ans = 3
                elif answer == "E":
                    ans = 4
                gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

                template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                gen_knowledge_template = gen_knowledge_template.format(question=question)
                knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                know_prompts_list = []
                for i in range(3):
                    know_prompts_list.append(knowledge_template)
                generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    
                template = self.prefix + template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred = generated_knowledge) + self.suffix + "Answer:"
                pred = llm_f(template)
                # process the prediction
                final_ans = self.text_processor(pred[0]['generated_text'])
                if final_ans == ans:
                    self.predictions.append(final_ans)
                    self.references.append(ans)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
                
            if self.task == "iwslt":
                # extract english text and answer in french
                eng_text = item['translation']['en']
                answer  = item['translation']['fr']
                gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                gen_knowledge_template = gen_knowledge_template.format(eng_text=eng_text)
                knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                know_prompts_list = []
                for i in range(3):
                    know_prompts_list.append(knowledge_template)
                generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                # create the final prompt and chain using llm_f
                template = self.prefix + template.format(eng_text=eng_text, pred = generated_knowledge) + self.suffix + "French:"
                pred = llm_f(template)
                # process the prediction
                final_ans = self.text_processor(pred[0]['generated_text'])
                bleu_score  = self.metrics[0]
                eval_score = bleu_score([final_ans],[answer])
                if  eval_score >= self.thres:
                    self.predictions.append(final_ans)
                    self.references.append(answer)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text'] 
            if self.task == "samsum":
                # extract the dialogue and summary
                dialogue = item['dialogue']
                answer = item['summary']
                gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                gen_knowledge_template = gen_knowledge_template.format(dialogue=dialogue)
                knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                know_prompts_list = []
                for i in range(3):
                    know_prompts_list.append(knowledge_template)
                generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                # create the final prompt and chain using llm_f
                template = self.prefix + template.format(dialogue=dialogue, pred = generated_knowledge) + self.suffix + "Summary:"
                pred = llm_f(template)
                # process the prediction
                final_ans = self.text_processor(pred[0]['generated_text'])
                rogue_score  = self.metrics[0]
                eval_score = rogue_score([final_ans],[answer])
                if  eval_score["rouge1"] >= self.thres:
                    self.predictions.append(final_ans)
                    self.references.append(answer)
                    return level, pred[0]['generated_text']
                else:
                    return 0, pred[0]['generated_text']
                    

    def process_dataset(self):
        '''
        processes the entire dataset using hierarchical prompts
        '''
        for item in self.dataset:
            prev = ""
            i = 0
            limit = 5
            if self.task == "boolq":
                answer = item['answer']
                if answer == True:
                    ans = 1 
                else:
                    ans = 0
            if self.task == "csqa":
                answer = item['answerKey']
                if answer == "A":
                    ans = 0
                elif answer == "B":
                    ans = 1
                elif answer == "C":
                    ans = 2
                elif answer == "D":
                    ans = 3
                elif answer == "E":
                    ans = 4
            if self.task == "iwslt":
                answer = item['translation']['fr']
                ans = answer
            if self.task == "samsum":
                answer = item['summary']
                ans = answer
            while i<limit:
                i = i+1
                level = self.select_prompt_level(item,prev)
                if level == 0 or level > 5:
                    continue
                llm_level, prev = self.prompt_process(item,level)
                if llm_level == level:
                    self.scores.append(level + i)
                    break
                if llm_level ==0:
                    pred= prev
                    prev = "The model was unable to solve the task at this level of prompting. The previous response was: " + prev + "\n"
                    continue
            if i == limit and level!=0:
                self.scores.append(i + hp_scores[self.task])
                final_ans = self.text_processor(pred)
                self.predictions.append(final_ans)
                self.references.append(ans)
        logging.info("***Dataset processed successfully***")
    
    def compute_scores(self):
        '''
        computes the scores for the predictions
        '''
        hp_score  = sum(self.scores)/len(self.scores)
        if self.task == "boolq":
            acc = self.metrics[0](self.predictions,self.references)
            f1 = self.metrics[1](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "accuracy": acc,
                "f1": f1
            }
            return scores
        elif self.task == "csqa":
            acc = self.metrics[0](self.predictions,self.references)
            f1 = self.metrics[1](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "accuracy": acc,
                "f1": f1
            }
            return scores
        elif self.task == "iwslt":
            bleu = self.metrics[0](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "bleu": bleu
            }
            return scores
        elif self.task == "samsum":
            rouge = self.metrics[0](self.predictions,self.references)
            scores = {
                "hp_score": hp_score,
                "rouge": rouge
            }
            return scores


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
        prefix = "<|user|>\n"
        suffix = "<|end|>\n<|assistant|>\n"
    elif model_name == "mistral":
        model = Mistral()
        prefix = "<s>[INST]\n"
        suffix = "[/INST]\n"
    elif model_name == "nemo":
        model = Nemo()
        prefix = "<s>[INST]\n"
        suffix = "[/INST]\n"
    elif model_name == "gemma2":
        model = Gemma2()
        prefix = "<bos><start_of_turn>user\n"
        suffix = "<end_of_turn>\n<start_of_turn>model\n"
    elif model_name == "gpt4o":
        model = GPT4o()
        prefix = "<|startoftext|>### Instruction: "
        suffix = "<|endoftext|>\n"
    elif model_name == "claude":
        model = Claude()
        prefix = "<|startoftext|>### Instruction: "
        suffix = "<|endoftext|>\n"
    
    dataset_name = args.arg3
    if dataset_name in ["iwslt", "samsum"]:
        thres = args.thres
    else :
        thres = 0

    data_loader = DatasetLoader()
    dataset = data_loader.get_dataset(dataset_name)
    text_processor = AnswerProcessor(dataset_name).processor
    eval_list  = Eval(dataset_name).metric

    if model_name == "llama3":
        gen_model = model
    else :
        gen_model = LLama3()

    if HP_framework == "man":
        manual_hp = ManualHierarchicalPrompt( model, gen_model, dataset, eval_list, text_processor, prompts, dataset_name, prefix, suffix,thres)
        logging.info("***Processing dataset using manual hierarchical prompt framework***")
        manual_hp.process_dataset()
        scores = manual_hp.compute_scores()
        with open("results.txt", "a") as file:
            if dataset_name in ["iwslt", "samsum"]:
                file.write("Dataset:" + dataset_name + "\n" + "Model:" + model_name + "\n" + "Threshold:" + str(thres) + "\n" + json.dumps(scores, indent=4) + "\n")
            else :
                file.write("Dataset:" + dataset_name + "\n" + "Model:" + model_name + "\n" + json.dumps(scores, indent=4) + "\n")


    elif HP_framework == "auto":
        adaptive_hp = AdaptiveHierarchicalPrompt(model, gen_model, dataset, eval_list, text_processor, prompts, dataset_name, prefix, suffix,thres)
        logging.info("***Processing dataset using adaptive hierarchical prompt framework***")
        adaptive_hp.process_dataset()
        scores = adaptive_hp.compute_scores()
        with open("results_adaptive.txt", "a") as file:
            if dataset_name in ["iwslt", "samsum"]:
                file.write("Mode=Adaptive\nDataset:" + dataset_name + "\n" + "Model:" + model_name + "\n" + "Threshold:" + str(thres) + "\n" + json.dumps(scores, indent=4) + "\n")
            else :
                file.write("Mode=Adaptive\nDataset:" + dataset_name + "\n" + "Model:" + model_name + "\n" + json.dumps(scores, indent=4) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for hierarchical prompt generation.')
    parser.add_argument('arg1', type=str, help='Manual or automatic prompt generation.')
    parser.add_argument('arg2', type=str, help='Model to be used.')
    parser.add_argument('arg3', type=str, help='Dataset to be used.')
    parser.add_argument('--thres', type=float , help='Threshold needed for iwslt and samsum datasets', default=0)
    args = parser.parse_args()
    main(args)