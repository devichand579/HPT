<div align="center">
  <div style="display: flex; align-items: center; justify-content: center;">
    <a href="https://github.com/devichand579/HPT">
      <img src="imgs/hpt_logo.png" alt="Logo" height="125">
    </a>
    <span style="font-size: 225px; font-weight: bold; margin-left: 10px;">
      <h1><strong>Hierarchical Prompting Taxonomy</strong></h1>
    </span>
  </div>
  <div style="margin-top: 10px;">
    <a href="">Paper</a>
    ·
    <a href="">Documentation</a>
    ·
    <a href="">Leaderboard</a>
  </div>
  <div style="font-size: 1.5em; font-weight: bold; margin-top: 10px;">
    A Universal Evaluation Framework for Large Language Models
  </div>
</div>

## News
## Introduction
## Demo
## Installation
## Usage
## Datasets and models 
### Datasets
HPT currently supports different datasets, models and prompt engineering methods employed by HPF. You are welcome to add more.


- Question-answering datasets:
  - BoolQ
- Reasoning datasets:
  - CommonsenseQA
- Translation datasets:
  - IWSLT-2017
- Summarization datasets:
  - SamSum


### Models

- Language models:
  - Llama 3 8B
  - Mistral 7B
  - Phi 3 3.8B
  - Gemma 7B


### Prompt Engineering

- Role Prompting
- Zero-shot Chain-of-Thought Prompting
- Three-shot Chain-of-Thought Prompting
- Least-to-Most Prompting
- Generated Knowledge Prompting
## Benchmark Results
## References 
## Contributing 
## Cite Us










## Setup Commands

To get started on a linux setup, follow these setup commands:


1. **Activate your conda environment:**
    ```sh
    conda activate <your_environment_name>
    ```
    
2. **Navigate to the main codebase**
   ```sh
   cd hierarchical_prompt
   ```
   
3. **Install the dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Adding your Hugging Face token**
   - Create a .env file
   ```sh
   HF_TOKEN = "your HF Token"
   ```

## Running the Framework

To run both the frameworks, use the following command structure:

```sh
bash run.sh method model dataset [--thres num]
```
method

  - man
  - auto
    
model

  - llama3
  - phi3
  - gemma
  - mistral
    
dataset

  - boolq
  - csqa
  - iwslt
  - samsum
    
If the dataset are IWSLT or SamSum, add '--thres num'

num
  - 0.15
  - 0.20
    
Example: 
   ```sh
   bash run.sh man llama3 iwslt --thres 0.15
   ```

