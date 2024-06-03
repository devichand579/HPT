from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import os
import numpy as np
from abc import ABC
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv('HF_TOKEN')


 
class Model(ABC):
    def __init__(self,name = None):
        self.model_names = {
            "llama3":   "meta-llama/Meta-Llama-3-8B-Instruct",
            "phi3": "microsoft/Phi-3-small-8k-instruct",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
            "gemma": "google/gemma-1.1-7b-it"
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)
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

                                model=self.model,
                                tokenizer=self.tokenizer,
                                eos_token_id=self.terminators,
                                do_sample=True,
                                return_full_text=False,
                                generation_config=self.generation_config
                               )
        
    def generate_knowledge(self,prompts):

        scores =[]
        preds = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
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
    
    def generate_pipe_f(self):
        llm_f = HuggingFacePipeline(self.pipe_f)
        return llm_f
    
    def generate_pipe_nf(self):
        llm_nf = HuggingFacePipeline(self.pipe_nf)
        return llm_nf
    
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
        
    
    def generate_pipe_f(self):
        llm_f = HuggingFacePipeline(self.pipe_f)
        return llm_f
    
    def generate_pipe_nf(self):
        llm_nf = HuggingFacePipeline(self.pipe_nf)
        return llm_nf
    
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
        
    
    def generate_pipe_f(self):
        llm_f = HuggingFacePipeline(self.pipe_f)
        return llm_f
    
    def generate_pipe_nf(self):
        llm_nf = HuggingFacePipeline(self.pipe_nf)
        return llm_nf
    
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
        
    
    def generate_pipe_f(self):
        llm_f = HuggingFacePipeline(self.pipe_f)
        return llm_f
    
    def generate_pipe_nf(self):
        llm_nf = HuggingFacePipeline(self.pipe_nf)
        return llm_nf
