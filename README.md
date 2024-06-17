<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!-- 
***[![License][license-shield]][license-url]
-->









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

