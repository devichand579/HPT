<div align="center">
  <div style="display: inline-flex; align-items: center;">
    <a href="https://github.com/devichand579/HPT">
      <img src="imgs/hpt_logo.png" alt="Logo" height="100">
    </a>
    <span style="font-size: 100px; font-weight: bold; line-height: 1; margin-left: 10px;">
      Hierarchical Prompting Taxonomy
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





## Dataset

The datasets used in this framework are:

- **BoolQ**: A dataset for yes/no questions.
- **CommonsenseQA**: A dataset for commonsense reasoning.
- **IWSLT2017**: A dataset for multilingual translation.
- **SAMsum**: A dataset for summarization of conversations.
  The HP-Scores and scoring policy of the corresponding datasets scored by Human experts can be found in [HP-Scores](./HP_scores) directory.

## Models Used

The following instruction-tuned models are utilized in this framework:

- **Llama 3 8B**
- **Mistral 7B**
- **Phi 3 3.8B**
- **Gemma 7B**

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

