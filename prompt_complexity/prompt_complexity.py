from prompts import Roleprompt, ZeroshotCoT, threeshotCoT, Leasttomost, generatedknowledge, Promptloader
import argparse
import json
import logging
import re
import random
import time
import os
from datasets import load_dataset
from openai import OpenAI
from abc import ABC
from pydantic import BaseModel
gpt_api_key = os.getenv('OPENAI_API_KEY')
prompts = {
    1 : Roleprompt(),
    2 : ZeroshotCoT(),
    3 : threeshotCoT(),
    4 : Leasttomost(),
    5 : generatedknowledge()
}


class Scores(BaseModel):
    score1 : int
    score2 : int
    score3 : int
    score4 : int


class GPT4o(ABC):
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o-2024-08-06"
        self.generation_config = {
            "max_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.90,
            "frequency_penalty": 1.15,
        }
        logging.info("***GPT-4O prompt level evaluation pipeline created successfully***")


    @property
    def pipe_nf(self):
        def generate(prompt):
            attempt = 0
            base_delay = 30  
            while True:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }],
                        **self.generation_config
                    )
                    pred = []
                    text = {"generated_text": completion.choices[0].message.content}
                    pred.append(text)
                    return pred
                
                except Exception as e:
                    attempt += 1
                    logging.warning(f"Exception occurred: {str(e)}. Attempt {attempt}...")
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Retrying after {delay:.2f} seconds...")
                    time.sleep(delay)

        return generate

    @property
    def pipe_scores(self):
        def generate(prompt):
            attempt = 0
            base_delay = 30  
            while True:
                try:
                    completion = self.client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": "you are a judge evaluating different prompting strategies and you need to score these prompting strategies based on a pre-defined criteria. Different prompting strategies leverage varied amount of intelligence from the model to achieve the required answer. So, assign the scores very carefully based on your analysis of the prompt and it's effect on your intelligence to achieve the given answer as well as the number of muilt-step prompts which increases the complexity of execution.\n score1: Basic Recall and Reproduction; Definition:The need of the model to remember and reproduce factual information without interpretation or analysis to answer the prompt; Range: 1-5\nscore2:Understanding and Interpretation; Definition: The need of the model to comprehend and explain the meaning of information, summarizing or clarifying content to answer the prompt; Range: 1-5\nscore3: Analysis and Reasoning; Definition: The need of the model to break down complex information, understand relationships, and solve problems using logical reasoning to answer the prompt; Range: 1-5\nscore4: Application of Knowledge and Execution; Definition: The need of the model to apply knowledge in practical situations, execute multi-step processes, and solve complex tasks to answer the prompt; Range: 1-5\n"},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=Scores,
                )
                    scores=completion.choices[0].message.parsed
                    return scores
                
                except Exception as e:
                    attempt += 1
                    logging.warning(f"Exception occurred: {str(e)}. Attempt {attempt}...")
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Retrying after {delay:.2f} seconds...")
                    time.sleep(delay)


        return generate


def main():
    mmlu_dataset = load_dataset('json', data_files='./data/mmlu.json')
    humaneval_dataset = load_dataset('json', data_files='./data/humaneval.json')
    gsm8k_dataset = load_dataset('json', data_files='./data/gsm8k.json')
    boolq_dataset = load_dataset('json', data_files='./data/boolq.json')
    csqa_dataset = load_dataset('json', data_files='./data/csqa.json')
    iwslt_dataset = load_dataset('json', data_files='./data/iwslt.json')
    samsum_dataset = load_dataset('json', data_files='./data/samsum.json')

    dataset_list = [mmlu_dataset, humaneval_dataset, gsm8k_dataset, boolq_dataset, csqa_dataset, iwslt_dataset, samsum_dataset]


    model = GPT4o()
    for i in range(1, 6):
        score1_list=[]
        score2_list=[]
        score3_list=[]
        score4_list=[]
        for process_dataset in dataset_list:
            for item in process_dataset['train']:
                if item['dataset'] == "mmlu":
                    task = "mmlu"
                    question = item['question']

                    text1 = item['choices'][0]
                    text2 = item['choices'][1]
                    text3 = item['choices'][2]
                    text4 = item['choices'][3]
                
                    ans = item['answer']
                    
                    if i==4:
                        templates = prompts[i].get_prompt(task)
                        pred_text = ""
                        for i in range(len(templates)):
                            prompt = templates[i].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, pred=pred_text)
                            if i != len(templates)-1:
                                pred = model.pipe_nf(prompt)
                                pred_text = pred[0]['generated_text']
                            else:
                                    scores = model.pipe_scores(prompt + "\n" + "Answer: " + ans + "\n" + "Multi-step prompts: 3")
                                    score1_list.append(scores.score1)
                                    score2_list.append(scores.score2)
                                    score3_list.append(scores.score3)
                                    score4_list.append(scores.score4)
                    elif i==5:

                        template, gen_knowledge_template = prompts[i].get_prompt(task)
                        gen_knowledge_template = gen_knowledge_template.format(question=question)
                        generated_knowledge = model.pipe_nf(gen_knowledge_template)
                        prompt = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, pred=generated_knowledge[0]['generated_text'])
                        scores  = model.pipe_scores(prompt + "\n" + "Answer: " + ans + "\n" + "Multi-step prompts: 1 (for external knowledge)")
                        score1_list.append(scores.score1)
                        score2_list.append(scores.score2)
                        score3_list.append(scores.score3)
                        score4_list.append(scores.score4)
                    else :
                        template = prompts[i].get_prompt(task)
                        prompt = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4)
                        scores = model.pipe_scores(prompt + "\n" + "Answer: " + ans + "\n" + "Multi-step prompts: 0")
                        score1_list.append(scores.score1)
                        score2_list.append(scores.score2)
                        score3_list.append(scores.score3)
                        score4_list.append(scores.score4)
                
                if item['dataset'] == "humaneval":
                    task = "humaneval"
                    code = item['prompt']
                    test_case = item['canonical_solution']
                    
                    if i==4:
                        templates = prompts[i].get_prompt(task)
                        pred_text = ""
                        for i in range(len(templates)):
                            prompt = templates[i].format(code=code, pred=pred_text)
                            if i != len(templates)-1:
                                pred = model.pipe_nf(prompt)
                                pred_text = pred[0]['generated_text']
                            else:
                                    scores = model.pipe_scores(prompt + "\n" + "Answer: " + test_case  +  "\n" + "Multi-step prompts: 3")
                                    score1_list.append(scores.score1)
                                    score2_list.append(scores.score2)
                                    score3_list.append(scores.score3)
                                    score4_list.append(scores.score4)
                    elif i==5:
                            template, gen_knowledge_template = prompts[i].get_prompt(task)
                            gen_knowledge_template = gen_knowledge_template.format(code=code)
                            generated_knowledge = model.pipe_nf(gen_knowledge_template)
                            prompt = template.format(code=code, pred=generated_knowledge[0]['generated_text'])
                            scores  = model.pipe_scores(prompt + "\n" + "Answer: " + test_case + "\n" + "Multi-step prompts: 1 (for external knowledge)")
                            score1_list.append(scores.score1)
                            score2_list.append(scores.score2)
                            score3_list.append(scores.score3)
                            score4_list.append(scores.score4)

                    else:
                        template = prompts[i].get_prompt(task)
                        prompt = template.format(code=code)
                        scores = model.pipe_scores(prompt + "\n" + "Answer: " + test_case + "\n" + "Multi-step prompts: 0")
                        score1_list.append(scores.score1)
                        score2_list.append(scores.score2)
                        score3_list.append(scores.score3)
                        score4_list.append(scores.score4)
            
                if item['dataset'] == "gsm8k":
                    task = "gsm8k"
                    question = item['question']
                    answer = item['answer'].split('#### ')[-1].strip()

                    if i==4:
                        templates = prompts[i].get_prompt(task)
                        pred_text = ""
                        for i in range(len(templates)):
                            prompt = templates[i].format(question=question, pred=pred_text)
                            if i != len(templates)-1:
                                pred = model.pipe_nf(prompt)
                                pred_text = pred[0]['generated_text']
                            else:
                                scores = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 3")
                                score1_list.append(scores.score1)
                                score2_list.append(scores.score2)
                                score3_list.append(scores.score3)
                                score4_list.append(scores.score4)

                    elif i==5:
                            template, gen_knowledge_template = prompts[i].get_prompt(task)
                            gen_knowledge_template = gen_knowledge_template.format(question=question)
                            generated_knowledge = model.pipe_nf(gen_knowledge_template)
                            prompt = template.format(question=question, pred=generated_knowledge[0]['generated_text'])
                            scores  = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 1 (for external knowledge)")
                            score1_list.append(scores.score1)
                            score2_list.append(scores.score2)
                            score3_list.append(scores.score3)
                            score4_list.append(scores.score4)

                    else:
                        template = prompts[i].get_prompt(task)
                        prompt = template.format(question=question)
                        scores = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 0")
                        score1_list.append(scores.score1)
                        score2_list.append(scores.score2)
                        score3_list.append(scores.score3)
                        score4_list.append(scores.score4)
                
                if item['dataset'] == "boolq":
                    task = "boolq"
                    passage = item['passage']
                    question = item['question']
                    answer = item['answer']
                    if answer == True:
                        ans = "True"
                    else:
                        ans = "False"
                    
                    if i==4:
                            templates = prompts[i].get_prompt(task)
                            pred_text = ""
                            for i in range(len(templates)):
                                prompt = templates[i].format(passage=passage, question=question, pred=pred_text)
                                if i != len(templates)-1:
                                    pred = model.pipe_nf(prompt)
                                    pred_text = pred[0]['generated_text']
                                else:
                                    scores = model.pipe_scores(prompt + "\n" + "Answer: " + ans + "\n" + "Multi-step prompts: 3")
                                    score1_list.append(scores.score1)
                                    score2_list.append(scores.score2)
                                    score3_list.append(scores.score3)
                                    score4_list.append(scores.score4)
                    
                    elif i==5:
                            template, gen_knowledge_template = prompts[i].get_prompt(task)
                            gen_knowledge_template = gen_knowledge_template.format(passage=passage, question=question)
                            generated_knowledge = model.pipe_nf(gen_knowledge_template)
                            prompt = template.format(passage=passage, question=question, pred=generated_knowledge[0]['generated_text'])
                            scores  = model.pipe_scores(prompt + "\n" + "Answer: " + ans + "\n" + "Multi-step prompts: 1 (for external knowledge)")
                            score1_list.append(scores.score1)
                            score2_list.append(scores.score2)
                            score3_list.append(scores.score3)
                            score4_list.append(scores.score4)

                    else:
                        template = prompts[i].get_prompt(task)
                        prompt = template.format(passage=passage, question=question)
                        scores = model.pipe_scores(prompt + "\n" + "Answer: " + ans + "\n" + "Multi-step prompts: 0")
                        score1_list.append(scores.score1)
                        score2_list.append(scores.score2)
                        score3_list.append(scores.score3)
                        score4_list.append(scores.score4)

                if item['dataset'] == "csqa":
                    task = "csqa"
                    question = item['question']
                    text1 = item['choices']['text'][0]
                    text2 = item['choices']['text'][1]
                    text3 = item['choices']['text'][2]
                    text4 = item['choices']['text'][3]
                    text5 = item['choices']['text'][4]
                    answer = item['answerKey']

                    if i==4:
                            templates = prompts[i].get_prompt(task)
                            pred_text = ""
                            for i in range(len(templates)):
                                prompt = templates[i].format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred=pred_text)
                                if i != len(templates)-1:
                                    pred = model.pipe_nf(prompt)
                                    pred_text = pred[0]['generated_text']
                                else:
                                    scores = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 3")
                                    score1_list.append(scores.score1)
                                    score2_list.append(scores.score2)
                                    score3_list.append(scores.score3)
                                    score4_list.append(scores.score4)
                    
                    elif i==5:
                            template, gen_knowledge_template = prompts[i].get_prompt(task)
                            gen_knowledge_template = gen_knowledge_template.format(question=question)
                            generated_knowledge = model.pipe_nf(gen_knowledge_template)
                            prompt = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5, pred=generated_knowledge[0]['generated_text'])
                            scores  = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 1 (for external knowledge)")
                            score1_list.append(scores.score1)
                            score2_list.append(scores.score2)
                            score3_list.append(scores.score3)
                            score4_list.append(scores.score4)

                    else:
                        template = prompts[i].get_prompt(task)
                        prompt = template.format(question=question, text1=text1, text2=text2, text3=text3, text4=text4, text5=text5)
                        scores = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 0")
                        score1_list.append(scores.score1)
                        score2_list.append(scores.score2)
                        score3_list.append(scores.score3)
                        score4_list.append(scores.score4)


                if item['dataset'] == "iwslt":
                    task = "iwslt"
                    eng_text = item['translation']['en']
                    answer  = item['translation']['fr']
                    
                    if i==4:
                            templates = prompts[i].get_prompt(task)
                            pred_text = ""
                            for i in range(len(templates)):
                                prompt = templates[i].format(eng_text=eng_text, pred=pred_text)
                                if i != len(templates)-1:
                                    pred = model.pipe_nf(prompt)
                                    pred_text = pred[0]['generated_text']
                                else:
                                    scores = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 3")
                                    score1_list.append(scores.score1)
                                    score2_list.append(scores.score2)
                                    score3_list.append(scores.score3)
                                    score4_list.append(scores.score4)

                    elif i==5:
                            template, gen_knowledge_template = prompts[i].get_prompt(task)
                            gen_knowledge_template = gen_knowledge_template.format(eng_text=eng_text)
                            generated_knowledge = model.pipe_nf(gen_knowledge_template)
                            prompt = template.format(eng_text=eng_text, pred=generated_knowledge[0]['generated_text'])
                            scores  = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 1 (for external knowledge)")
                            score1_list.append(scores.score1)
                            score2_list.append(scores.score2)
                            score3_list.append(scores.score3)
                            score4_list.append(scores.score4)

                    else:
                        template = prompts[i].get_prompt(task)
                        prompt = template.format(eng_text=eng_text)
                        scores = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 0")
                        score1_list.append(scores.score1)
                        score2_list.append(scores.score2)
                        score3_list.append(scores.score3)
                        score4_list.append(scores.score4)

                if item['dataset'] == "samsum":
                    task = "samsum"
                    dialogue = item['dialogue']
                    answer = item['summary']

                    if i==4:
                            templates = prompts[i].get_prompt(task)
                            pred_text = ""
                            for i in range(len(templates)):
                                prompt = templates[i].format(dialogue=dialogue, pred=pred_text)
                                if i != len(templates)-1:
                                    pred = model.pipe_nf(prompt)
                                    pred_text = pred[0]['generated_text']
                                else:
                                    scores = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 3")
                                    score1_list.append(scores.score1)
                                    score2_list.append(scores.score2)
                                    score3_list.append(scores.score3)
                                    score4_list.append(scores.score4)

                    elif i==5:
                            template, gen_knowledge_template = prompts[i].get_prompt(task)
                            gen_knowledge_template = gen_knowledge_template.format(dialogue=dialogue)
                            generated_knowledge = model.pipe_nf(gen_knowledge_template)
                            prompt = template.format(dialogue=dialogue, pred=generated_knowledge[0]['generated_text'])
                            scores  = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 1 (for external knowledge)")
                            score1_list.append(scores.score1)
                            score2_list.append(scores.score2)
                            score3_list.append(scores.score3)
                            score4_list.append(scores.score4)
                    
                    else:
                        template = prompts[i].get_prompt(task)
                        prompt = template.format(dialogue=dialogue)
                        scores = model.pipe_scores(prompt + "\n" + "Answer: " + answer + "\n" + "Multi-step prompts: 0")
                        score1_list.append(scores.score1)
                        score2_list.append(scores.score2)
                        score3_list.append(scores.score3)
                        score4_list.append(scores.score4)
            
        avg1 = sum(score1_list) / len(score1_list)
        avg2 = sum(score2_list) / len(score2_list)
        avg3 = sum(score3_list) / len(score3_list)
        avg4 = sum(score4_list) / len(score4_list)
        avg_total = (avg1 + avg2 + avg3 + avg4) / 4
        if i==1:
            prompt_strategy = "Role Prompting"
        elif i==2:
            prompt_strategy = "Zero-shot CoT Prompting"
        elif i==3:
            prompt_strategy = "Three-shot CoT Prompting"
        elif i==4:
            prompt_strategy = "Least to Most Prompting"
        else:
            prompt_strategy = "Generated Knowledge Prompting" 
            
        with open('scores.json', 'a') as f:
            json.dump({'prompt strategy':prompt_strategy, "score1": avg1, "score2": avg2, "score3": avg3, "score4": avg4, "total": avg_total}, f, indent=4)
        print(f"Prompt Strategy: {prompt_strategy} written to scores.json")

if __name__ == "__main__":
    main()