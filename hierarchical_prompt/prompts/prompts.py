from abc import ABC
"""
   An utility class to load various prompts across different tasks for the hierarchical prompt framework.
"""
class Promptloader(ABC):
    def __init__(self):
       self.prompts = {
           "boolq": None,
           "csqa": None,
           "iwslt": None,
           "samsum": None,
           "gsm8k": None,
           "humaneval": None
       }

       self.generate_knowledge_prompts = {
              "boolq": ("Generate Knowledge about the passage: {0}").format("{passage}"),
                "csqa": ("Generate Knowledge about the question: {0}").format("{question}"),
                "iwslt": ("Generate definitions in french of each word in the text: {0}").format("{eng_text}"),
                "samsum": ("Generate interpretation about the dialogue: {0}").format("{dialogue}"),
                "gsm8k": ("Generate Knowledge about the question: {0}").format("{question}"),
                "humaneval": ("Generate interpretation about the code: {0}").format("{code}"),
                "mmlu": ("Generate Knowledge about the question: {0}").format("{question}")
        }
       
       
       self.adaptive_prompt = ("{0}Choose the most effective prompting strategy among five prompting strategies for the task. Start with the least indexed prompting strategy which is most effective and move to higher indexed prompting strategies if lower level prompting strategies are not effective\nTask:{1}\nThe prompting strategies are:\n1: Role Prompting -  Defines a role for the model in solving the task.\n2: Zero-shot Chain of Thought prompting - stimulate reasoning and problem-solving by including the phrase 'Let's think step by step' without offering previous examples related to the task.\n3: Three-shot Chain of Thought prompting -  Offers three examples related to the task to guide the model's reasoning process.\n4: Least-to-most prompting  -  Uses a sequential method to derive essential insights from the task in order to solve it.\n5: Generated Knowledge Prompting - Integration and application of external knowledge to accomplish the task. The external knowledge is generated using some other model based on the task.\nSelect only the index(Don't provide the name) of the most effective prompting strategy.").format("{prev_res}","{task}")


class Roleprompt(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = ("Based on the passage:'{0}'\nAnswer True/False to the question: '{1}' as an Omniscent person.").format("{passage}", "{question}")
        self.prompts["csqa"] = ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4}\nE {5} as a critical thinker.").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{text5}")
        self.prompts["iwslt"] = ("Translate '{0}' to french as a Translator.").format("{eng_text}")
        self.prompts["samsum"] = ("Summarise the Dialogue: '{0}' as a Summariser.").format("{dialogue}")  
        self.prompts["gsm8k"] = ("Based on the question:'{0}'\nCalculate the numerical answer to the question as a expert mathematician.").format( "{question}")
        self.prompts["humaneval"] = ("Complete the given code based on the mentioned constraints: {0} as an expert programmer.").format("{code}")
        self.prompts["mmlu"] = ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4} as a critical thinker.").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}")

    def get_prompt(self, task):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        else :
            return self.prompts[task]
        
class ZeroshotCoT(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = ("Based on the passage:'{0}'\nAnswer True/False to the question: '{1}'.\nLet's think step by step.").format("{passage}", "{question}")
        self.prompts["csqa"] = ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4}\nE {5}\nLet's think step by step.").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{text5}")
        self.prompts["iwslt"] = ("Translate '{0}' to french.\nLet's think step by step.").format("{eng_text}")
        self.prompts["samsum"] = ("Summarise the Dialogue: '{0}'.\nLet's think step by step.").format("{dialogue}")   
        self.prompts["gsm8k"] = ("Based on the question:'{0}'\nCalculate the numerical answer to the question.\nLet's think step by step.").format( "{question}")
        self.prompts["humaneval"] = ("Complete the given code based on the mentioned constraints: {0}\nLet's think step by step.").format("{code}")
        self.prompts["mmlu"] = ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4}\nLet's think step by step.").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}")
    def get_prompt(self, task):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        else :
            return self.prompts[task]

class threeshotCoT(Promptloader):
    def __init__(self):
        super().__init__()
        passage1 = "Harry Potter and the Escape from Gringotts is an indoor steel roller coaster at Universal Studios Florida, a theme park located within the Universal Orlando Resort. Similar to dark rides, the roller coaster utilizes special effects in a controlled-lighting environment and also employs motion-based 3-D projection of both animation and live-action sequences to enhance the experience. The ride, which is themed to the Gringotts Wizarding Bank, became the flagship attraction for the expanded Wizarding World of Harry Potter when it opened on July 8, 2014."
        passage2 = "All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline."
        passage3 = "The Commonwealth government has its own tax laws and Puerto Ricans are also required to pay some US federal taxes, although most residents do not have to pay the federal personal income tax. In 2009, Puerto Rico paid $3.742 billion into the US Treasury. Residents of Puerto Rico pay into Social Security, and are thus eligible for Social Security benefits upon retirement. However, they are excluded from the Supplemental Security Income."
        question1 = "is harry potter and the escape from gringotts a roller coaster ride"
        question2 = "does ethanol take more energy make that produces"
        question3 = "is federal income tax the same as social security"
        ans1 = "true"
        ans2 = "false"
        ans3 = "false"
        exp1 = "Harry Potter and the Escape from Gringotts is an indoor steel roller coaster at Universal Studios Florida, a theme park located within the Universal Orlando Resort."
        exp2 = "The total amount of energy input into the process for producing ethanol compared to the energy released by burning the resulting ethanol fuel is known as the energy balance, a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. another survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. Both surveys suggest that ethanol produces more energy than it takes to produce ethanol."
        exp3 = "Puerto Ricans are also required to pay some US federal taxes, although most residents do not have to pay the federal personal income tax. Residents of Puerto Rico pay into Social Security which is different from federal income tax and provides benefits upon retirement."
        
        question_1 = "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"
        question_2 = "Where would you find magazines along side many other printed works?"
        question_3 = "What would vinyl be an odd thing to replace?"
        exp_1 = "Revolving doors are common in buildings with high foot traffic because they allow people to enter and exit simultaneously without creating drafts or requiring doors to be constantly opened and closed. In the context of a bank, security is a paramount concern. Revolving doors can serve as a controlled access point, making it harder for unauthorized individuals to enter or exit quickly. While libraries, department stores, and malls might also use revolving doors for the convenience of two-way travel, the specific mention of security measures aligns best with the stringent security requirements of a bank."
        exp_2 = "Bookstores are establishments that sell books and other printed materials. They are known for their wide selection of reading materials, including magazines, newspapers, and periodicals. While markets and train stations may also have printed materials available for purchase, bookstores are specifically designed to cater to readers and book enthusiasts. The mention of magazines alongside other printed works suggests a location that specializes in reading materials, making a bookstore the most appropriate choice."
        exp_3 = "Wallpaper is a type of material used to cover and decorate the interior walls of homes, offices, and other buildings. It is typically sold in rolls and comes in a variety of patterns and designs. While pants, record albums, and cheese are all items that can be purchased or consumed, they are not typically associated with a record store. A record store is a retail establishment that specializes in selling music recordings, such as vinyl records, CDs, and cassettes. The mention of wallpaper alongside other items suggests that it is an odd choice for a vinyl, making it the correct answer."
       
        ex_en1 = "This drying around the world has lead to a dramatic increase in fires."
        ex_en2 = "Look carefully at the area of the eastern Pacific, from the Americas, extending westward, and on either side of the Indian subcontinent, where there is a radical depletion of oxygen in the oceans."
        ex_en3 = "If you look at in the context of history you can see what this is doing."
        ex_fr1 = "Cet assèchement global a causé une hausse spectaculaire du nombre d'incendies. Cet assèchement global a causé une hausse spectaculaire du nombre d'incendies."
        ex_fr2 = "Regardez le secteur oriental de l'océan pacifique depuis les Amériques vers l'ouest, et des deux côtés du sous-continent Indien, la raréfaction de l'oxygène y est dramatique."
        ex_fr3 = "Si vous la remettez dans son contexte, vous pouvez voir à quoi ressemble cette tendance."
        
        ex_dialogue1 = "Antonio: Is everything okay? You've been quiet lately Alivia: Oh, hi, yeah, I've just been working on my thesis Alivia: Or rather trying to work, it's not going too well Antonio: Oh :( Problems finding research materials? Alivia: Well Alivia: That isn't really as big a problem, the worst part is actually sitting down and writing Alivia: I find the topic interesting and all, I don't mind reading articles and books Alivia: But when I'm supposed to write, it's like I blank out and can't type a single word w/o thinking I sound stupid... Antonio: I know the feeling... Antonio: You should probably stop thinking about it so seriously, just write and you can edit it later Antonio: Once you get past the initial difficulty, it'll get better, at least that's what it was like for me Alivia: I'd like to think so... Thanks... I'll try. And thanks for your concern <3"
        ex_dialogue2 = "Maddie: I'm in Asda, do you need anything? John: could do with a white bread and some apples Maddie: ok. Gala? John: yes please ta"
        ex_dialogue3 = "Rob: hey, pick up your phone :) Ann: can't - meeting :) Rob: sorry... Ann: no problem - super boring one :) Ann: what you need babe? Rob: I'm at the grocery store and was wondering if we need anything Ann: some food :) Rob: yeah, I figured that smartass :) Ann: :* Rob: details? so that you won't moan we don't have anything to eat :) Ann: from what I remember we have everything for supper and lunch tomorrow, maybe some fruit and vegetables? Rob: anything in particular? Ann: cucumber, tomatoes, bananas, apples and whatever you like Rob: ok"
        ex_sum1 = "Alivia has been taciturn lately. She was trying to write her thesis. She can't focus on writing. She'll try to follow Antonio's advice to start writing without overthinking."
        ex_sum2 = "Maddie will buy a white bread and apples on John's request."
        ex_sum3 = "Rob is doing shopping at the grocery store. Ann ordered him to buy a cucumber, some tomatoes, bananas and apples."

        gsm8k_question1 = "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
        gsm8k_question2 = "Vincent can buy flowers in packages of 3 for $2.50 or in packages of 2 for $1. How much money does he save by buying 18 flowers at the better price?"
        gsm8k_question3 = "In a neighborhood, the number of rabbits pets is twelve less than the combined number of pet dogs and cats. If there are two cats for every dog, and the number of dogs is 60, how many pets in total are in the neighborhood?"
        gsm8k_ans1 = "70000\nThe cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000 He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000 So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000 So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000 #### 70000"
        gsm8k_ans2 = "6\nFind how many packages of 3 would be needed which is 18 ÷ 3 = <<18/3=6>>6. The cost of using packages of 3 is 6 × $2.50 = $<<6*2.5=15>>15. Find how many packages of 2 would be needed which is 18 ÷ 2 = <<18/2=9>>9. The cost of using packages of 2 is 9 × $1 = $<<9*1=9>>9. Vincent would save $15 - $9 = $<<15-9=6>>6. #### 6"
        gsm8k_ans3 = "348\nIf there are two cats for every dog, and the number of dogs is 60, the number of cats is 2*60 = <<2*60=120>>120 The combined number of cats and dogs is 120+60 = <<120+60=180>>180 The number of rabbits pets is twelve less than the combined number of pet dogs and cats, a total of 180-12 = 168 The total number of pets in the compound is 168+180 = <<168+180=348>>348 #### 348"

        humaneval_code1 = '''from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """'''
        humaneval_code2 = '''def is_prime(n):
    """Return true if a given number is prime, and false otherwise.
    >>> is_prime(6)
    False
    >>> is_prime(101)
    True
    >>> is_prime(11)
    True
    >>> is_prime(13441)
    True
    >>> is_prime(61)
    True
    >>> is_prime(4)
    False
    >>> is_prime(1)
    False
    """'''
        humaneval_code3 = '''def triangle_area(a, b, c):
    """Given the lengths of the three sides of a triangle. Return the area of the triangle rounded to 2 decimal points if the three sides form a valid triangle. Otherwise return -1
    Three sides make a valid triangle when the sum of any two sides is greater than the third side.
    Example:
    triangle_area(3, 4, 5) == 6.00
    triangle_area(1, 2, 10) == -1
    """'''
        humaneval_sol1 = "for idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = abs(elem - elem2) if distance < threshold: return True return False"
        humaneval_sol2 = "if n < 2: return False for k in range(2, n - 1): if n % k == 0: return False return True"
        humaneval_sol3 = "if a + b <= c or a + c <= b or b + c <= a: return -1 s = (a + b + c)/2 area = (s * (s - a) * (s - b) * (s - c)) ** 0.5 area = round(area, 2) return area"


        mmlu_ques1 = "Any set of Boolean operators that is sufficient to represent all Boolean expressions is said to be complete. Which of the following is NOT complete?"
        mmlu_ques2 = "The accused made a confession to the police, but his defense attorney needs to find out if the confession is admissible. In which of the following situations would a confession be inadmissible?"
        mmlu_ques3 = "For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | When I went through the road block I gave the policeman my brothers ID. Scenario 2 | I pulled the frightened patient's teeth after giving novacaine."
       
        mmlu_exp1 = "The set {AND, OR} is not complete because it cannot represent all possible Boolean functions on its own. Specifically, it lacks the NOT operator, which is necessary to invert any Boolean value. Without the ability to perform negation, you cannot create expressions like XOR or NAND, making the set insufficient for full Boolean expression representation."
        mmlu_exp2 = "Involuntary confessions are inadmissible because they are obtained through coercion, threats, or improper inducement, violating the defendant's rights and undermining the integrity of the legal process. For a confession to be admissible, it must be made voluntarily, with the defendant's awareness of their rights, and during a proper custodial interrogation."
        mmlu_exp3 = "In Scenario 1, giving a policeman someone else's ID to evade a roadblock is morally wrong because it involves deceit and dishonesty, which violates ethical standards and legal norms. This act undermines trust in legal processes and can lead to legal consequences. In contrast, Scenario 2 involves pulling a patient's teeth under novocaine, which, if done by a qualified professional with informed consent, is a standard medical procedure and not considered morally wrong."

        self.prompts["boolq"] = ("Based on the passage:'{0}'\nAnswer True/False to the question: '{1}'.\nAnswer: {2}.\nExplaination: {3}.\nBased on the passage:'{4}'\nAnswer True/False to the question: '{5}'.\nAnswer: {6}.\nExplaination: {7}.\nBased on the passage:'{8}'\nAnswer True/False to the question: '{9}'.\nAnswer: {10}.\nExplaination: {11}.\nBased on the passage:'{12}'\nAnswer True/False to the question: '{13}'").format( passage1, question1, ans1, exp1, passage2, question2, ans2, exp2,  passage3, question3, ans3, exp3, "{passage}", "{question}" )
        self.prompts["csqa"] = ("Choose the answer.\n{0}\nA bank\nB library\nC department store\nD mall\nE new york\nAnswer: A\nExplanation: {1}.\nChoose the answer.\n{2}\nA doctor\nB bookstore\nC market\nD train station\nE mortuary\nAnswer: B\nExplanation: {3}.\nChoose the answer.\n{4}\nA pants\nB record albums\nC record store\nD cheese\nE wallpaper\nAnswer: E\nExplanation: {5}.\nChoose the answer.\n{6}\nA {7}\nB {8}\nC {9}\nD {10}\nE {11}").format( question_1, exp_1, question_2, exp_2, question_3, exp_3, "{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{text5}" )
        self.prompts["iwslt"] = ("Translate '{0}' to French.\nFrench: {1}.\nTranslate '{2}' to French.\nFrench: {3}.\nTranslate '{4}' to French.\nFrench: {5}.\nTranslate '{6}' to French.").format( ex_en1, ex_fr1, ex_en2, ex_fr2, ex_en3, ex_fr3, "{eng_text}" )
        self.prompts["samsum"] = ("Summarise the Dialogue: '{0}'.\nSummary: {1}.\nSummarise the Dialogue: '{2}'.\nSummary: {3}.\nSummarise the Dialogue: '{4}'.\nSummary: {5}.\nSummarise the Dialogue: '{6}'").format( ex_dialogue1, ex_sum1, ex_dialogue2, ex_sum2, ex_dialogue3, ex_sum3, "{dialogue}" )
        self.prompts["gsm8k"] = ("Based on the question:'{0}'\nCalculate the numerical answer to the question.\nAnswer: {1}.\nBased on the question:'{2}'\nCalculate the numerical answer to the question.\nAnswer: {3}.\nBased on the question:'{4}'\nCalculate the numerical answer to the question.\nAnswer: {5}\nBasedBased on the question:'{6}'\nCalculate the numerical answer to the question.").format( gsm8k_question1, gsm8k_ans1, gsm8k_question2, gsm8k_ans2, gsm8k_question3, gsm8k_ans3, "{question}" )
        self.prompts["humaneval"] = ("Complete the given code based on the mentioned constraints: {0}\nCode: {1}.\nComplete the given code based on the mentioned constraints: {2}\nCode: {3}.\nComplete the given code based on the mentioned constraints: {4}\nCode: {5}.\nComplete the given code based on the mentioned constraints: {6}\nCode: {7}").format( humaneval_code1, humaneval_sol1, humaneval_code2, humaneval_sol2, humaneval_code3, humaneval_sol3, "{code}" )
        self.prompts["mmlu"] = ("Choose the answer.\n{0}\nA {AND, NOT}\nB {NOT, OR}\nC {AND, OR}\nD {NAND}\nAnswer: C\nExplanation: '{1}'.\nChoose the answer.\n{2}\nA The defendant's statement was involuntary.\nB The defendant's statement was voluntary.\nC The defendant was not in custody when the statement was made.\nD The statement was not made in response to a known police interrogation.\nAnswer: A\nExplanation: '{3}'.\nChoose the answer.\n{4}\nA Wrong, Wrong\nB Wrong, Not wrong\nC Not wrong, Wrong\nD Not wrong, Not wrong\nAnswer: B\nExplanation: '{5}'.\nChoose the answer.\n{6}\nA {7}\nB {8}\nC {9}\nD {10}").format( mmlu_ques1, mmlu_exp1, mmlu_ques2, mmlu_exp2, mmlu_ques3, mmlu_exp3, "{question}", "{text1}", "{text2}", "{text3}", "{text4}" )
    def get_prompt(self, task):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        else :
            return self.prompts[task]
        
class Leasttomost(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = [("Summarize the main points of this passage: '{0}'").format("{passage}"),("Analyze this question to identify its key components: '{0}'").format("{question}"),("Find the part of the passage that relates to this question: '{0}'\nPassage: '{1}'").format("{question}","{passage}"),("Based on the passage, what is the answer to this question: '{0}'\nRelevant Information: '{1}'").format("{question}","{pred}")]
        self.prompts["csqa"] = [("Analyze this question: '{0}'").format("{question}"), ("Elaborate about each option for the question: '{0}'\noptions: A {1}\nB {2}\nC {3}\nD {4}\nE {5}").format("{question}","{text1}","{text2}","{text3}","{text4}","{text5}"),("Based on the analysis : '{0}', Discard wrong answers among the options:A {1}\nB {2}\nC {3}\nD {4}\nE {5}").format("{pred}", "{text1}","{text2}","{text3}","{text4}","{text5}"), ("Choose the correct answer from the options: A {0}\nB {1}\nC {2}\nD {3}\nE {4}").format("{text1}","{text2}","{text3}","{text4}","{text5}")]
        self.prompts["iwslt"] = [("What is the main idea or theme of this text? '{0}'").format("{eng_text}"),("Identify and list the key phrases or terms in this text: '{0}'").format("{eng_text}"),("Translate the following key phrases into french: '{0}'").format("{pred}"),("Translate '{0}' into french, incorporating the translations of the key phrases: '{1}'").format("{eng_text}","{pred}")]
        self.prompts["samsum"] = [("List the main points or key ideas present in this dialogue: '{0}'.").format("{dialogue}"),("Elaborate on the following key points, providing additional details or context: '{0}'.").format("{pred}"),("Using the listed key points and their elaborations, draft a concise summary of this text: '{0}'.").format("{dialogue}"),("Refine this draft summary to make it more concise and coherent, ensuring it captures the essence of the text: '{0}'.").format("{dialogue}")]
        self.prompts["gsm8k"] = [("Analyze the question: '{0}'").format("{question}"),("Break the question into sub problems: '{0}'").format("{question}"),("Calulate answers for the subproblems of the question: '{0}'").format("{pred}"),("Calculate the numerical answer to this question: '{0}' based on the previous calculations: '{1}'").format("{question}","{pred}")]
        self.prompts["humaneval"] = [("Analyze the code: '{0}'").format("{code}"),("Break the problem into sub problems: '{0}'").format("{code}"),("Complete code for the subproblems of the code: '{0}'").format("{pred}"),("Complete the code based on mentioned constraints: '{0}' based on the previous calculations: '{1}'").format("{code}","{pred}")]
        self.prompts["mmlu"] = [("Analyze this question: '{0}'").format("{question}"), ("Elaborate about each option for the question: '{0}'\noptions: A {1}\nB {2}\nC {3}\nD {4}").format("{question}","{text1}","{text2}","{text3}","{text4}"),("Based on the analysis : '{0}', Discard wrong answers among the options:A {1}\nB {2}\nC {3}\nD {4}").format("{pred}", "{text1}","{text2}","{text3}","{text4}"), ("Choose the correct answer from the options: A {0}\nB {1}\nC {2}\nD {3}").format("{text1}","{text2}","{text3}","{text4}")]


    def get_prompt(self, task):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        else :
            return self.prompts[task]
        
class generatedknowledge(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = ("Based on the passage:'{0}'\nAnswer True/False to the question: '{1}' using knowledge of the passage:{2}.").format("{passage}", "{question}", "{pred}")
        self.prompts["csqa"] = ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4}\nE {5} using knowledge of the question:{6}").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{text5}", "{pred}")
        self.prompts["iwslt"] = ("Translate '{0}' to french using definitions of the keywords:{1}").format("{eng_text}", "{pred}")
        self.prompts["samsum"] = ("Summarise the Dialogue: '{0}' using interpretation of the dialogue:{1}").format("{dialogue}", "{pred}")
        self.prompts["gsm8k"] = ("Based on the question:'{0}'\nCalculate the numerical answer to the question using interpretation of the question:{1}").format( "{question}", "{pred}")
        self.prompts["humaneval"] = ("Complete the code based on the mentioned constraints:{0} using knowledge of the constraints:{1}").format("{code}", "{pred}")
        self.prompts["mmlu"] = ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4} using knowledge of the question:{5}").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{pred}")

    def get_prompt(self, task):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        else :
            return self.prompts[task], self.generate_knowledge_prompts[task]