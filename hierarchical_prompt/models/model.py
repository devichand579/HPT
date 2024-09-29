from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch
import os
import numpy as np
from abc import ABC
import logging 
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
import requests
import time
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
gpt_api_key = os.getenv('OPENAI_API_KEY')
claude_api_key = os.getenv('ANTHROPIC_API_KEY')


 
class Model(ABC):
    def __init__(self,name = None):
        self.model_names = {
            "llama3":   "meta-llama/Meta-Llama-3-8B-Instruct",
            "phi3": "microsoft/Phi-3-mini-4k-instruct",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
            "gemma": "google/gemma-1.1-7b-it",
            "nemo": "mistralai/Mistral-Nemo-Instruct-2407",
            "gemma2": "google/gemma-2-9b-it"
        }
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = None
        self.tokenizer = None
        self.pipe_nf = None
        self.pipe_f = None
        self.generation_config = None
        self.load_model(name)

    def load_model(self,name):
        model_name = self.model_names.get(name)
        if model_name is None:
            raise ValueError(f"Model name '{name}' is not supported.")
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=self.quantization_config, 
                token = hf_token
                )
        logging.info(f"***Model {name} loaded successfully***")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)
        logging.info(f"***Tokenizer for {name} loaded successfully***")
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.generation_config.max_new_tokens = 1024 
        self.generation_config.temperature = 0.6 
        self.generation_config.top_p = 0.90 
        self.generation_config.do_sample = True 
        self.generation_config.repetition_penalty = 1.15 

class LLama3(Model):
    def __init__(self):
        super().__init__("llama3")
        self.terminators = [
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                          ]
        self.pipe_f = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                eos_token_id=self.terminators,
                                do_sample=True,
                                return_full_text=True,
                                generation_config=self.generation_config
                               )
        self.pipe_nf = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                eos_token_id=self.terminators,
                                do_sample=True,
                                return_full_text=False,
                                generation_config=self.generation_config
                               )
        logging.info("***LLama3 text generation pipelines created successfully***")
        
    def generate_knowledge(self,prompts):

        scores =[]
        preds = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            input_ids = input_ids.to(self.model.device)
            outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            repetition_penalty=1.15,
            temperature = 0.6,
            top_p = 0.90,
            return_dict_in_generate=True,
            output_scores=True,
            )
            gen_sequences = outputs.sequences[:, input_ids.shape[-1]:]
            probs = torch.stack(outputs.scores, dim=1).softmax(-1)
            gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            score = torch.mean(gen_probs)
            preds.append(self.tokenizer.decode(gen_sequences[0]))
            scores.append(score.numpy())


        scores = np.array(scores)
        final_idx = np.argmax(scores)
        return preds[final_idx]
    

class Phi3(Model):
    def __init__(self):
        super().__init__("phi3")
        

        self.pipe_f = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=True,
            return_full_text=True,
            generation_config=self.generation_config
        )
        self.pipe_nf = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=True,
            return_full_text=False,
            generation_config=self.generation_config
        )

        logging.info("*Phi3 text generation pipelines created successfully*")

        
    
class Mistral(Model):
    def __init__(self):
        super().__init__("mistral")
        self.pipe_f = pipeline(
                               "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                do_sample=True,
                                return_full_text=True,
                                generation_config=self.generation_config
                               )
        self.pipe_nf = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                do_sample=True,
                                return_full_text=False,
                                generation_config=self.generation_config
                               )

        logging.info("***Mistral text generation pipelines created successfully***")
        
 
    
class Gemma(Model):
    def __init__(self):
        super().__init__("gemma")
        self.model.bfloat16()
        self.pipe_f = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                do_sample=True,
                                return_full_text=True,
                                generation_config=self.generation_config
                               )
        self.pipe_nf = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                do_sample=True,
                                return_full_text=False,
                                generation_config=self.generation_config
                               )

        logging.info("***Gemma text generation pipelines created successfully***")
   
class Nemo(Model):
    def __init__(self):
        super().__init__("nemo")
        self.pipe_f = pipeline(
                               "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                do_sample=True,
                                return_full_text=True,
                                generation_config=self.generation_config
                               )
        self.pipe_nf = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                do_sample=True,
                                return_full_text=False,
                                generation_config=self.generation_config
                               )

        logging.info("***Mistral-Nemo text generation pipelines created successfully***")


class Gemma2(Model):
    def __init__(self):
        super().__init__("gemma2")
        self.model.bfloat16()
        self.pipe_f = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                do_sample=True,
                                return_full_text=True,
                                generation_config=self.generation_config
                               )
        self.pipe_nf = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                do_sample=True,
                                return_full_text=False,
                                generation_config=self.generation_config
                               )

        logging.info("***Gemma-2-9B text generation pipelines created successfully***")
    

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
        self.initial_retries = 3 
        self.extended_retries = 1  
        self.retry_delay = 65  
        logging.info("***GPT-4O text generation pipelines created successfully***")

    @property
    def pipe_f(self):
        def generate(prompt):
            for attempt in range(self.initial_retries + self.extended_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Provide the answer at the start of first sentence"
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        **self.generation_config
                    )
                    pred = []
                    text = {"generated_text": prompt + completion.choices[0].message.content}
                    pred.append(text)
                    return pred
                except Exception as e:
                    if attempt < self.initial_retries-1:
                        logging.warning(f"Exception occurred: {str(e)}. Quick retry {attempt + 1}/{self.initial_retries}...")
                    elif attempt == self.initial_retries - 1:
                        logging.warning(f"Initial retries failed. Waiting {self.retry_delay} seconds before extended retry...")
                        time.sleep(self.retry_delay)
                    elif attempt == self.initial_retries:
                        logging.warning(f"Initial retries failed. Waiting 1 hour before extended retry...")
                        time.sleep(3600)
                    else:
                        logging.error(f"All retries failed. Last exception: {str(e)}")
                        raise e  # Raise the last exception after all retries

        return generate

    @property
    def pipe_nf(self):
        def generate(prompt):
            for attempt in range(self.initial_retries + self.extended_retries):
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
                    if attempt < self.initial_retries-1:
                        logging.warning(f"Exception occurred: {str(e)}. Quick retry {attempt + 1}/{self.initial_retries}...")
                    elif attempt == self.initial_retries - 1:
                        logging.warning(f"Initial retries failed. Waiting {self.retry_delay} seconds before extended retry...")
                        time.sleep(self.retry_delay)
                    elif attempt == self.initial_retries:
                        logging.warning(f"Initial retries failed. Waiting 1 hour before extended retry...")
                        time.sleep(3600)
                    else:
                        logging.error(f"All retries failed. Last exception: {str(e)}")
                        raise e  # Raise the last exception after all retries

        return generate


    
class Claude(ABC):
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-3-5-sonnet-20240620"
        self.initial_retries = 3 
        self.extended_retries = 1  
        self.retry_delay = 65 
        logging.info("***Claude text generation pipelines created successfully***")

    @property
    def pipe_f(self):
        def generate(prompt):
            for attempt in range(self.initial_retries + self.extended_retries):
                try:
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        temperature=0.6,
                        top_p=0.90,
                        system="Provide the answer at the start of first sentence",
                        messages=[
                            {
                                "role": "user",
                                "content": (prompt)
                            }
                        ]
                    )
                    pred = []
                    text = {"generated_text": prompt + message.content[0].text}
                    pred.append(text)
                    return pred
                except Exception as e:
                    if attempt < self.initial_retries-1:
                        logging.warning(f"Exception occurred: {str(e)}. Quick retry {attempt + 1}/{self.initial_retries}...")
                    elif attempt == self.initial_retries - 1:
                        logging.warning(f"Initial retries failed. Waiting {self.retry_delay} seconds before extended retry...")
                        time.sleep(self.retry_delay)
                    elif attempt == self.initial_retries:
                        logging.warning(f"Initial retries failed. Waiting 1 hour before extended retry...")
                        time.sleep(3600)
                    else:
                        logging.error(f"All retries failed. Last exception: {str(e)}")
                        raise e  # Raise the last exception after all retries

        return generate

    @property
    def pipe_nf(self):
        def generate(prompt):
            for attempt in range(self.initial_retries + self.extended_retries):
                try:
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        temperature=0.6,
                        top_p=0.90,
                        messages=[
                            {
                                "role": "user",
                                "content": (prompt)
                            }
                        ]
                    )
                    pred = []
                    text = {"generated_text": message.content[0].text}
                    pred.append(text)
                    return pred
                except Exception as e:
                    if attempt < self.initial_retries-1:
                        logging.warning(f"Exception occurred: {str(e)}. Quick retry {attempt + 1}/{self.initial_retries}...")
                    elif attempt == self.initial_retries-1:
                        logging.warning(f"Initial retries failed. Waiting 1 hour seconds before extended retry...")
                        time.sleep(self.retry_delay)
                    elif attempt == self.initial_retries:
                        logging.warning(f"Initial retries failed. Waiting 1 hour before extended retry...")
                        time.sleep(3600)
                    else:
                        logging.error(f"All retries failed. Last exception: {str(e)}")
                        raise e  # Raise the last exception after all retries


        return generate


