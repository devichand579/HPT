<p align="center">
    <a href="https://github.com/badges/shields/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/badges/shields" /></a>
    <a href="#backers" alt="Backers on Open Collective">
        <img src="https://img.shields.io/opencollective/backers/shields" /></a>
    <a href="#sponsors" alt="Sponsors on Open Collective">
        <img src="https://img.shields.io/opencollective/sponsors/shields" /></a>
    <a href="https://github.com/badges/shields/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/badges/shields" /></a>
    <a href="https://circleci.com/gh/badges/shields/tree/master">
        <img src="https://img.shields.io/circleci/project/github/badges/shields/master" alt="build status"></a>
    <a href="https://circleci.com/gh/badges/daily-tests">
        <img src="https://img.shields.io/circleci/project/github/badges/daily-tests?label=service%20tests"
            alt="service-test status"></a>
    <a href="https://coveralls.io/github/badges/shields">
        <img src="https://img.shields.io/coveralls/github/badges/shields"
            alt="coverage"></a>
    <a href="https://discord.gg/HjJCwm5">
        <img src="https://img.shields.io/discord/308323056592486420?logo=discord"
            alt="chat on Discord"></a>
</p>

# Hierarchical Prompting Taxonomy: A Universal Evaluation Framework for Large Language Models

This repository contains the implementation and dataset related to the paper *Hierarchical Prompting Taxonomy: A Universal Evaluation Framework for Large Language Models*.

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

