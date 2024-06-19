<a name="readme-top"></a>
<!-- <p align="center">
  <img src="https://img.shields.io/github/stars/devichand579/HPT" alt="GitHub stars" height="25"/>
  <img src="https://img.shields.io/github/forks/devichand579/HPT" alt="GitHub forks" height="25"/>
  <img src="https://img.shields.io/github/contributors/devichand579/HPT" alt="GitHub contributors" height="25"/>
  <img src="https://img.shields.io/github/issues/devichand579/HPT" alt="GitHub issues" height="25"/>
</p> -->

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
    <a href="https://arxiv.org/abs/2406.12644">Paper</a>
    ·
    <a href="">Documentation</a>
    ·
    <a href="">Leaderboard</a>
  </div>
  <div style="font-size: 1.5em; font-weight: bold; margin-top: 10px;">
    <strong>A Universal Evaluation Framework for Large Language Models</strong>
  </div>
</div>

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#news">News</a></li>
    <li><a href="#introduction">Introduction</a></li>
    <!-- <li><a href="#demo">Demo</a></li> -->
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#supported-datasets-and-models">Datasets and Models</a></li>
    <!-- <li><a href="#benchmark-results">Benchmark Results</a></li> -->
    <!-- <li><a href="#references">References</a></li> -->
    <li><a href="#contributing">Contributing</a></li> 
    <li><a href="#cite-us">Cite Us</a></li>
  </ol>
</details>

## News
[18-06-24] HPT is published ! Check out the paper [here](https://arxiv.org/abs/2406.12644).

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Introduction
**Hierarchical Prompting Taxonomy** (HPT) is a universal evaluation framework for large language models. It is designed to evaluate the performance of large language models on a variety of tasks and datasets assigning **HP-Score** for each dataset relative to different models. The HPT employs **Hierarchical Prompt Framework** (HPF) which supports a wide range of tasks, including question-answering, reasoning, translation, and summarization. It provides a set of pre-defined prompting strategies tailored for each task based on its complexity. Refer to paper at : [https://arxiv.org/abs/2406.12644](https://arxiv.org/abs/2406.12644)

![HPT](imgs/hpt.png)
<!-- ### Features of HPT and HPF -->
<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

<!-- ## Demo -->
<!-- Refer to [examples](./examples/) directory for using the framework on different datasets and models. -->
## Installation
<!-- ### pip install
To install the package, run the following command:
```sh
pip install hpt
``` -->
### Cloning the Repository
To clone the repository, run the following command:
```sh
git clone https://github.com/devichand579/HPT.git
```

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Usage
### Linux
To get started on a Linux setup, follow these setup commands:
1. **Activate your conda environment:**
    ```sh
    conda activate hpt
    ```
3. **Navigate to the main codebase**
   ```sh
   cd HPT/hierarchical_prompt
   ```
   
3. **Install the dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Add your Hugging Face token**
   - Create a .env file in the conda environment
   ```sh
   HF_TOKEN = "your HF Token"
   ```

5. **To run both frameworks, use the following command structure**
    ```sh
    bash run.sh method model dataset [--thres num]
    ```
    - method
      - man
      - auto
        
    - model
        - llama3
        - phi3
        - gemma
        - mistral
        
    - dataset
        - boolq
        - csqa
        - iwslt
        - samsum
        
     - If the datasets are IWSLT or SamSum, add '--thres num'

    - num
        - 0.15
        - 0.20
        
    - Example commands: 
      ```sh
      bash run.sh man llama3 iwslt --thres 0.15
      ```
      ```sh
      bash run.sh auto phi3 boolq 
      ```
<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Datasets and models 
HPT currently supports different datasets, models and prompt engineering methods employed by HPF. You are welcome to add more.
### Datasets

- Question-answering datasets:
  - BoolQ
- Reasoning datasets:
  - CommonsenseQA
- Translation datasets:
  - IWSLT-2017 en-fr
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

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Benchmark Results -->
## References 
## Contributing 
This project aims to build open-source evaluation frameworks for assessing LLMs and other agents. This project welcomes contributions and suggestions. Please see the details on [how to contribute](contribute.md).

If you are new to GitHub, [here](https://opensource.guide/how-to-contribute/#how-to-submit-a-contribution) is a detailed guide on getting involved with development on GitHub.
<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Cite Us
If you find our work useful, please cite us !
```bibtex
@misc{budagam2024hierarchical,
      title={Hierarchical Prompting Taxonomy: A Universal Evaluation Framework for Large Language Models}, 
      author={Devichand Budagam and Sankalp KJ and Ashutosh Kumar and Vinija Jain and Aman Chadha},
      year={2024},
      eprint={2406.12644},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>


