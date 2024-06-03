from dataloader import DatasetLoader
from metrics import Eval
from models import LLama3, Gemma, Phi3, Mistral
from prompts import Roleprompt, ZeroshotCoT, threeshotCoT, Leasttomost, generatedknowledge
from utils import AnswerProcessor
from abc import ABC
import argparse
import json
from langchain_community import HuggingFacePipeline
from langchain_core import PromptTemplate

#Prompts Dictionary
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
    "samsum": 2.23
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
        for i in range(1,6):
            llm_f = self.model.generate_pipe_f()    # full_text pipeline
            llm_nf = self.model.generate_pipe_nf()  # non_full_text pipeline

            # handles passage and ques-ans pairs
            if self.task == "boolq":
                #extracting the passage, question, and answer from the item
                passage = item['passage']
                question = item['question']
                answer = item['answer']
                if answer == "true":
                    ans = 1
                else:
                    ans = 0

                # level 4
                if i==4:
                    # retrieve multiple levels of least-to-most prompting
                    pred = ""
                    templates = self.prompts[i].get_prompt(self.task)
                    # iterate over the templates
                    for i in range(len(templates)):
                        # and create a prompt chain for each of them
                        template = self.prefix + templates[i].format(passage=passage, question=question, pred=pred) + self.suffix + "Answer:"
                        prompt = PromptTemplate.from_template(template)

                        # for intermediate templates, use llm_nf
                        if i != len(templates)-1:
                            chain = prompt | llm_nf
                            pred = chain.invoke({'question': question,'passage':passage, 'pred':pred})
                        # for final template, use llm_f
                        else:
                            chain = prompt | llm_f
                            pred = chain.invoke({'question': question,'passage':passage, 'pred':pred})

                    # process the prediction
                    final_ans = self.text_processor(pred)
                    if final_ans == ans:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    # retrieve the template and create a prompt chain using llm_nf
                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(passage=passage)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    # create the final prompt and chain using llm_f
                    template = self.prefix + template.format(passage=passage, question=question, pred = generated_knowledge) + self.suffix + "Answer:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    pred = chain.invoke({'question': question,'passage':passage,'pred':generated_knowledge})

                    # process the prediction
                    final_ans = self.text_processor(pred)
                    if final_ans == ans:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        i = i + hp_scores[self.task]
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                    
                
                # for other levels, retrieve the prompt template, add the prefix and suffix, and create a prompt chain using llm_f
                else :
                    template = self.prompts[i].get_prompt(self.task).format(passage=passage, question=question)
                    template = self.prefix + template + self.suffix +"Answer:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    pred = chain.invoke({'question': question,'passage':passage})
                    final_ans = self.text_processor(pred)
                    if final_ans == ans:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        continue

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
                    pred = ""
                    # iterate over the templates
                    for i in range(len(templates)):
                        # and create a prompt chain for each of them
                        template = self.prefix + templates[i].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred=pred) + self.suffix + "Answer:"
                        prompt = PromptTemplate.from_template(template)

                        # for intermediate templates, use llm_nf
                        if i != len(templates)-1:
                            chain = prompt | llm_nf
                            pred = chain.invoke({'question': question,'text1':text1,'text2':text2,'text3':text3,'text4':text4,'text5':text5,'pred':pred})
                        # for final template, use llm_f
                        else:
                            chain = prompt | llm_f
                            pred = chain.invoke({'question': question,'text1':text1,'text2':text2,'text3':text3,'text4':text4,'text5':text5,'pred':pred})

                    # process the prediction
                    final_ans = self.text_processor(pred)
                    if final_ans == ans:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    # retrieve the template and create a prompt chain using llm_nf
                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(question=question)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    # create the final prompt and chain using llm_f
                    template = self.prefix + template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred = generated_knowledge) + self.suffix + "Answer:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    pred = chain.invoke({'question': question,'text1':text1,'text2':text2,'text3':text3,'text4':text4,'text5':text5,'pred':generated_knowledge})

                    # process the prediction
                    final_ans = self.text_processor(pred)
                    if final_ans == ans:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        i = i + hp_scores[self.task]
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                    
                
                # for other levels, retrieve the prompt template, add the prefix and suffix, and create a prompt chain using llm_f
                else :
                    template = self.prompts[i].get_prompt(self.task).format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
                    template = self.prefix + template + self.suffix +"Answer:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    pred = chain.invoke({'question': question,'text1':text1,'text2':text2,'text3':text3,'text4':text4,'text5':text5})
                    final_ans = self.text_processor(pred)
                    if final_ans == ans:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(ans)
                        break
                    else:
                        continue
            # handles translation tasks
            elif self.task == "iwslt":
                # extract english text and answer in french
                eng_text = item['translation'][0]['en']
                answer  = item['translation']['fr']

                # level 4
                if i==4:
                    # retrieve multiple levels of least-to-most prompting
                    templates = self.prompts[i].get_prompt(self.task)
                    pred = ""
                    # iterate over the templates
                    for i in range(len(templates)):
                        

                        # for intermediate templates, use llm_nf
                        if i != len(templates)-1:
                            template = self.prefix + templates[i].format(eng_text=eng_text, pred=pred) + self.suffix
                            prompt = PromptTemplate.from_template(template)
                            chain = prompt | llm_nf
                            pred = chain.invoke({'text': eng_text,'pred':pred})
                        # for final template, use llm_f
                        else:
                            template = self.prefix + templates[i].format(eng_text=eng_text, pred=pred) + self.suffix + "French:"
                            prompt = PromptTemplate.from_template(template)
                            chain = prompt | llm_f
                            pred = chain.invoke({'text': eng_text,'pred':pred})

                    # process the prediction
                    final_ans = self.text_processor(pred)
                    bleu_score  = self.metrics[0]
                    eval_score = bleu_score(final_ans,answer)
                    if  eval_score >= self.thres:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    # retrieve the template and create a prompt chain using llm_nf
                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(eng_text=eng_text)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    # create the final prompt and chain using llm_f
                    template = self.prefix + template.format(eng_text=eng_text, pred = generated_knowledge) + self.suffix + "French:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    pred = chain.invoke({'eng_text': eng_text,'pred':generated_knowledge})

                    # process the prediction
                    final_ans = self.text_processor(pred)
                    bleu_score  = self.metrics[0]
                    eval_score = bleu_score(final_ans,answer)
                    if  eval_score >= self.thres:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        i = i + hp_scores[self.task]
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                
                
                # for other levels, retrieve the prompt template, add the prefix and suffix, and create a prompt chain using llm_f
                else :
                    template = self.prompts[i].get_prompt(self.task).format(eng_text=eng_text)
                    template = self.prefix + template + self.suffix +"French:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    pred = chain.invoke({'eng_text': eng_text})
                    final_ans = self.text_processor(pred)
                    bleu_score  = self.metrics[0]
                    eval_score = bleu_score(final_ans,answer)
                    if  eval_score >= self.thres:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
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
                    pred = ""
                    # iterate over the templates
                    for i in range(len(templates)):
                        

                        # for intermediate templates, use llm_nf
                        if i != len(templates)-1:
                            template = self.prefix + templates[i].format(dialogue=dialogue, pred=pred) + self.suffix
                            prompt = PromptTemplate.from_template(template)
                            chain = prompt | llm_nf
                            pred = chain.invoke({'dialogue': dialogue,'pred':pred})
                        # for final template, use llm_f
                        else:
                            template = self.prefix + templates[i].format(dialogue=dialogue, pred=pred) + self.suffix + "Summary:"
                            prompt = PromptTemplate.from_template(template)
                            chain = prompt | llm_f
                            pred = chain.invoke({'dialogue': dialogue,'pred':pred})

                    # process the prediction
                    final_ans = self.text_processor(pred)
                    rouge_score  = self.metrics[0]
                    eval_score = rouge_score(final_ans,answer)
                    if  eval_score >= self.thres:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        continue

                # level 5
                elif i==5:
                    gen_prefix = "<|start_header_id|>user<|end_header_id|>\n"
                    gen_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    # retrieve the template and create a prompt chain using llm_nf
                    template, gen_knowledge_template = self.prompts[i].get_prompt(self.task)
                    gen_knowledge_template = gen_knowledge_template.format(dialogue=dialogue)
                    knowledge_template = gen_prefix + gen_knowledge_template + gen_suffix
                    know_prompts_list = []
                    for i in range(3):
                        know_prompts_list.append(knowledge_template)
                    generated_knowledge = self.gen_model.generate_knowledge(know_prompts_list)
              
                    # create the final prompt and chain using llm_f
                    template = self.prefix + template.format(dialogue=dialogue, pred = generated_knowledge) + self.suffix + "Summary:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    pred = chain.invoke({'dialogue': dialogue,'pred':generated_knowledge})

                    # process the prediction
                    final_ans = self.text_processor(pred)
                    rouge_score  = self.metrics[0]
                    eval_score = rouge_score(final_ans,answer)
                    if  eval_score >= self.thres:
                        self.scores.append(i)
                        self.predictions.append(pred)
                        self.references.append(answer)
                        break
                    else:
                        i = i + hp_scores[self.task]
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                    
                
                # for other levels, retrieve the prompt template, add the prefix and suffix, and create a prompt chain using llm_f
                else :
                    template = self.prompts[i].get_prompt(self.task).format(dialogue=dialogue)
                    template = self.prefix + template + self.suffix + "Summary:"
                    prompt = PromptTemplate.from_template(template)
                    chain = prompt | llm_f
                    pred = chain.invoke({'dialogue': dialogue})
                    final_ans = self.text_processor(pred)
                    rouge_score  = self.metrics[0]
                    eval_score = rouge_score(final_ans,answer)
                    if  eval_score >= self.thres:
                        self.scores.append(i)
                        self.predictions.append(final_ans)
                        self.references.append(answer)
                        break
                    else:
                        continue
    def process_dataset(self):
        '''
        processes the entire dataset using hierarchical prompts
        '''
        for item in self.dataset:
            self.prompt_process(item)

    
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

            


# class AdaptiveHierarchicalPrompt(ABC):
#     def __init__(self, model, dataset, metric, text_processor, prompts, task, prefix, suffix):
#         self.model = model
#         self.dataset = dataset
#         self.metric = metric
#         self.text_processor = text_processor
#         self.prompts = prompts
#         self.task = task
#         self.prefix = prefix
    #     self.suffix = suffix

    # def select_prompt_level(self, item):
    #     # Use LLaMA3 or another model to choose the prompt level based on the item
    #     # Here, llm_selection_model is an instance of the model used for selecting the prompt level
    #     selection_prompt = f"Select the appropriate prompt level (1-5) for the following item: {item}"
    #     selected_level = self.model.select_prompt_level(selection_prompt)
    #     return int(selected_level.strip())

    # def prompt_process(self, item):
    #     selected_level = self.select_prompt_level(item)
    #     llm_f = self.model.generate_pipe_f()
    #     llm_nf = self.model.generate_pipe_nf()

    #     passage = item['passage']
    #     question = item['question']
    #     ans = item['answer']
        
    #     if self.task == "boolq":
    #         if selected_level == 1:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(passage=passage, question=question)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'question': question, 'passage': passage})

    #         elif selected_level == 2:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(passage=passage, question=question)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'question': question, 'passage': passage})

    #         elif selected_level == 3:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(passage=passage, question=question)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'question': question, 'passage': passage})

    #         elif selected_level == 4:
    #             templates = self.prompts[selected_level].get_prompt(self.task)
    #             predictions = ""
    #             for i in range(len(templates)):
    #                 prompt_text = templates[i].format(passage=passage, question=question, predictions=predictions)
    #                 prompt = PromptTemplate.from_template(prompt_text)
    #                 if i != len(templates) - 1:
    #                     chain = prompt | llm_nf
    #                 else:
    #                     chain = prompt | llm_f
    #                 predictions = chain.invoke({'question': question, 'passage': passage})

    #         elif selected_level == 5:
    #             template, gen_knowledge_prompt = self.prompts[selected_level].get_prompt(self.task)
    #             knowledge_prompt_text = gen_knowledge_prompt.format(passage=passage)
    #             generated_knowledge = llm_nf.invoke({'passage': passage, 'question': question})
    #             prompt_text = template.format(passage=passage, question=question, pred=generated_knowledge)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'question': question, 'passage': passage})

    #     if self.task == 'csqa':
    #         question = item['question']
    #         choices = item['choices']
    #         text1 = choices[0]['text']
    #         text2 = choices[1]['text']
    #         text3 = choices[2]['text']
    #         text4 = choices[3]['text']
    #         text5 = choices[4]['text']
    #         ans = item['answerKey']

    #         if selected_level == 1:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

    #         elif selected_level == 2:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

    #         elif selected_level == 3:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

    #         elif selected_level == 4:
    #             templates = self.prompts[selected_level].get_prompt(self.task)
    #             predictions = ""
    #             for i in range(len(templates)):
    #                 prompt_text = templates[i].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, predictions=predictions)
    #                 prompt = PromptTemplate.from_template(prompt_text)
    #                 if i != len(templates) - 1:
    #                     chain = prompt | llm_nf
    #                 else:
    #                     chain = prompt | llm_f
    #                 predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

    #         elif selected_level == 5:
    #             template, gen_knowledge_prompt = self.prompts[selected_level].get_prompt(self.task)
    #             knowledge_prompt_text = gen_knowledge_prompt.format(question=question)
    #             generated_knowledge = llm_nf.invoke({'question': question})
    #             prompt_text = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred=generated_knowledge)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'question': question, 'text1': text1, 'text2': text2, 'text3': text3, 'text4': text4, 'text5': text5})

    #     if self.task == 'iwslt':
    #         eng_text = item['translation']['en']
    #         fr_text = item['translation']['fr']

    #         if selected_level == 1:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(eng_text=eng_text)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'eng_text': eng_text})

    #         elif selected_level == 2:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(eng_text=eng_text)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'eng_text': eng_text})

    #         elif selected_level == 3:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(eng_text=eng_text)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'eng_text': eng_text})

    #         elif selected_level == 4:
    #             templates = self.prompts[selected_level].get_prompt(self.task)
    #             predictions = ""
    #             for i in range(len(templates)):
    #                 prompt_text = templates[i].format(eng_text=eng_text, predictions=predictions)
    #                 prompt = PromptTemplate.from_template(prompt_text)
    #                 if i != len(templates) - 1:
    #                     chain = prompt | llm_nf
    #                 else:
    #                     chain = prompt | llm_f
    #                 predictions = chain.invoke({'eng_text': eng_text})

    #         elif selected_level == 5:
    #             template, gen_knowledge_prompt = self.prompts[selected_level].get_prompt(self.task)
    #             knowledge_prompt_text = gen_knowledge_prompt.format(eng_text=eng_text)
    #             generated_knowledge = llm_nf.invoke({'eng_text': eng_text})
    #             prompt_text = template.format(eng_text=eng_text, pred=generated_knowledge)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'eng_text': eng_text})

    #     if self.task == 'samsum':
    #         dialogue = item['dialogue']
    #         summary = item['summary']

    #         if selected_level == 1:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(dialogue=dialogue)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'dialogue': dialogue})

    #         elif selected_level == 2:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(dialogue=dialogue)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'dialogue': dialogue})

    #         elif selected_level == 3:
    #             template = self.prompts[selected_level].get_prompt(self.task)
    #             prompt_text = template.format(dialogue=dialogue)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'dialogue': dialogue})

    #         elif selected_level == 4:
    #             templates = self.prompts[selected_level].get_prompt(self.task)
    #             predictions = ""
    #             for i in range(len(templates)):
    #                 prompt_text = templates[i].format(dialogue=dialogue, predictions=predictions)
    #                 prompt = PromptTemplate.from_template(prompt_text)
    #                 if i != len(templates) - 1:
    #                     chain = prompt | llm_nf
    #                 else:
    #                     chain = prompt | llm_f
    #                 predictions = chain.invoke({'dialogue': dialogue})

    #         elif selected_level == 5:
    #             template, gen_knowledge_prompt = self.prompts[selected_level].get_prompt(self.task)
    #             knowledge_prompt_text = gen_knowledge_prompt.format(dialogue=dialogue)
    #             generated_knowledge = llm_nf.invoke({'dialogue': dialogue})
    #             prompt_text = template.format(dialogue=dialogue, pred=generated_knowledge)
    #             prompt = PromptTemplate.from_template(prompt_text)
    #             chain = prompt | llm_f
    #             predictions = chain.invoke({'dialogue': dialogue})

    #     return predictions


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

    data_loader = DatasetLoader()
    dataset = data_loader.get_dataset(dataset_name)
    text_processor = AnswerProcessor(dataset_name).processor
    eval_list  = Eval(dataset_name).metric

    if model_name == "llama3":
        gen_model = model
    else :
        gen_model = LLama3()

    if HP_framework == "man":
        manual_hp = ManualHierarchicalPrompt(model, gen_model, dataset, eval_list, text_processor, prompts, dataset_name, prefix, suffix,thres)
        manual_hp.process_dataset()
        scores = manual_hp.compute_scores()
        with open("results.txt", "a") as file:
            if dataset_name in ["iwslt", "samsum"]:
                file.write("Dataset:" + dataset_name + "\n" + "Model:" + model_name + "\n" + "Threshold:" + thres + "\n" + json.dumps(scores, indent=4) + "\n")
            else :
                file.write("Dataset:" + dataset_name + "\n" + "Model:" + model_name + "\n" + json.dumps(scores, indent=4) + "\n")


    # elif HP_framework == "auto":
    #     adaptive_hp = AdaptiveHierarchicalPrompt(model, dataset, eval, text_processor, prompts, dataset_name, prefix, suffix)
    #     for item in dataset:
    #         adaptive_hp.prompt_process(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for hierarchical prompt generation.')
    parser.add_argument('arg1', type=str, help='Manual or automatic prompt generation.')
    parser.add_argument('arg2', type=str, help='Model to be used.')
    parser.add_argument('arg3', type=str, help='Dataset to be used.')
    parser.add_argument('--thres', type=int, help='Threshold needed for iwslt and samsum datasets', default=0)
    args = parser.parse_args()
    main(args)