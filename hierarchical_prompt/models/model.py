from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch
from langchain import HuggingFacePipeline


 
class Model:
    def __init__(self, name):
        self.model_names = {
            "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
            "phi3": "microsoft/Phi-3-small-8k-instruct",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
            "gemma": "google/gemma-1.1-7b-it"
        }
        self.name = name
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
        self.load_model()

    def load_model(self):
        model_name = self.model_names.get(self.name)
        if model_name is None:
            raise ValueError(f"Model name '{self.name}' is not supported.")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe_nf = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.generation_config
        )
        self.pipe_f = HuggingFacePipeline(pipeline=self.pipe_nf)

    def set_generation_config(self, **kwargs):
        self.generation_config = GenerationConfig(**kwargs)

    def generate_text(self, prompt):
        output = self.pipe_nf(prompt)
        return output[0]['generated_text']
