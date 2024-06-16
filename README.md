# Hierarchical Prompting Taxonomy: A Universal Evaluation Framework for Large Language Models

This repository contains the implementation and dataset related to the paper *Hierarchical Prompting Taxonomy: A Universal Evaluation Framework for Large Language Models*.

## Dataset

The datasets used in this framework are:

- **BoolQ**: A dataset for yes/no questions.
- **CommonsenseQA**: A dataset for commonsense reasoning.
- **IWSLT2017**: A dataset for multilingual translation.
- **SAMsum**: A dataset for summarization of conversations.

## Models Used

The following models are utilized in this framework:

- **Llama 3 8B**
- **Mistral 7B**
- **Phi 3 3.8B**
- **Gemma 7B**

## Setup Commands

To get started, follow these setup commands:

1. **Start the SSH agent in the background:**
    ```sh
    eval "$(ssh-agent -s)"
    ```

2. **Add your SSH private key to the SSH agent:**
    ```sh
    ssh-add ./hp
    ```

3. **Activate your conda environment:**
    ```sh
    conda activate <your_environment_name>
    ```
4. **Install the dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Framework

To run the framework, use the following command structure:

```sh
method model dataset [--thres num]
