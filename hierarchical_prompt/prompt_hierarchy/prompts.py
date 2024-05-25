from abc import ABC

class Promptloader(ABC):
    def __init__(self):
       self.prompts = {
           "boolq": None,
           "csqa": None,
           "iwslt": None,
           "samsum": None
       }
       self.interelation_prompts = {
           "boolq": "The response for the previous prompt which has resulted in a wrong answer: '{response}'",
           "csqa": "The response for the previous prompt which has resulted in a wrong answer: '{response}'",
           "iwslt": "The response for the previous prompt which was unable to cross the evaluation threshold: '{response}'",
           "samsum": "The response for the previous prompt which was unable to cross the evaluation threshold: '{response}'"
       }


class Roleprompting(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = "Based on the passage:'{passage}'\nAnswer True/False to the question: '{question}' as an Omniscient person."
        self.prompts["csqa"] = "Choose the answer as a critical thinker.\n{question}\n{opt1}. {text1}\n{opt2}. {text2}\n{opt3}. {text3}\n{opt4}. {text4}\n{opt5}. {text5}"
        self.prompts["iwslt"] = "Translate '{eng_text}' to french as a Translator."
        self.prompts["samsum"] = "Summarise the Dialogue: {dialogue} as a Storyteller."   

    def get_prompt(self, task,interelation=False):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        if interelation:
            return self.interelation_prompts[task], self.prompts[task]
        else :
            return self.prompts[task]
        
class ZeroshotCoT(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = "Based on the passage:'{passage}'\nAnswer True/False to the question: '{question}'.Let's think step by step."
        self.prompts["csqa"] = "Choose the answer.\n{question}\n{opt1}. {text1}\n{opt2}. {text2}\n{opt3}. {text3}\n{opt4}. {text4}\n{opt5}. {text5}\nLet's think step by step."
        self.prompts["iwslt"] = "Translate '{eng_text}' to french.Let's translate step by step."
        self.prompts["samsum"] = "Summarise the Dialogue: {dialogue}.Let's summarise step by step."   

    def get_prompt(self, task,interelation=False):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        if interelation:
            return self.interelation_prompts[task], self.prompts[task]
        else :
            return self.prompts[task]

class threeshotCoT(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = "Based on the passage:'{passage1}'\nAnswer True/False to the question: '{question1}'.\nAnswer: '{ans1}.\nExplaination: '{exp1}'.\nBased on the passage:'{passage2}'\nAnswer True/False to the question: '{question2}'.\nAnswer: '{ans2}.\nExplaination: '{exp2}'.\nBased on the passage:'{passage3}'\nAnswer True/False to the question: '{question3}'.\nAnswer: '{ans3}.\nExplaination: '{exp3}'.\nBased on the passage:'{passage}'\nAnswer True/False to the question: '{question}'"
        self.prompts["csqa"] = "Choose the answer.\n{question_1}\n{opt1_1}. {text1_1}\n{opt2_1}. {text2_1}\n{opt3_1}. {text3_1}\n{opt4_1}. {text4_1}\n{opt5_1}. {text5_1}\nAnswer: {ans1}.\nExplaination: {exp1}.Choose the answer.\n{question_2}\n{opt1_2}. {text1_2}\n{opt2_2}. {text2_2}\n{opt3_2}. {text3_2}\n{opt4_2}. {text4_2}\n{opt5_2}. {text5_2}\nAnswer: {ans2}.\nExplaination: {exp2}.\nChoose the answer.\n{question_3}\n{opt1_3}. {text1_3}\n{opt2_3}. {text2_3}\n{opt3_3}. {text3_3}\n{opt4_3}. {text4_3}\n{opt5_3}. {text5_3}\nAnswer: {ans3}.\nExplaination: {exp3}.\nChoose the answer.\n{question}\n{opt1}. {text1}\n{opt2}. {text2}\n{opt3}. {text3}\n{opt4}. {text4}\n{opt5}. {text5}"
        self.prompts["iwslt"] = "Translate '{ex_en1}' to french.\nFrench: {ex_fr1}.\nTranslate '{ex_en2}' to french.\nFrench: {ex_fr2}.\nTranslate '{ex_en3}' to french.\nFrench: {ex_fr3}.\nTranslate '{eng_text}' to french."
        self.prompts["samsum"] = "Summarise the Dialogue: {ex_dialogue1}.\nSummary: {ex_sum1}.\nSummarise the Dialogue: {ex_dialogue2}.\nSummary: {ex_sum2}.\nSummarise the Dialogue: {ex_dialogue3}.\nSummary: {ex_sum3}.\nSummarise the Dialogue: {dialogue}"   

    def get_prompt(self, task,interelation=False):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        if interelation:
            return self.interelation_prompts[task], self.prompts[task]
        else :
            return self.prompts[task]
        
class Leasttomost(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = ""
        self.prompts["csqa"] = ""
        self.prompts["iwslt"] = ""  
        self.prompts["samsum"] = ""

    def get_prompt(self, task,interelation=False):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        if interelation:
            return self.interelation_prompts[task], self.prompts[task]
        else :
            return self.prompts[task]
        
class generatedknowledge(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = ""
        self.prompts["csqa"] = ""
        self.prompts["iwslt"] = ""  
        self.prompts["samsum"] = ""
 

    def get_prompt(self, task,interelation=False):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        if interelation:
            return self.interelation_prompts[task], self.prompts[task]
        else :
            return self.prompts[task]