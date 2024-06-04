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
           "samsum": None
       }

       self.generate_knowledge_prompts = {
              "boolq": ("Generate Knowledge about the passage: {0}").format("{passage}"),
                "csqa": ("Generate Knowledge about the question: {0}").format("{question}"),
                "iwslt": ("Generate definitions in french of each word in the text: {0}").format("{eng_text}"),
                "samsum": ("Generate interpretation about the dialogue: {0}").format("{dialogue}")
        }
       
       self.adaptive_prompt = ""


class Roleprompt(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = ("Based on the passage:'{0}'\nAnswer True/False to the question: '{1}' as an Omniscent person.").format("{passage}", "{question}")
        self.prompts["csqa"] = ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4}\nE {5} as a critical thinker.").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{text5}")
        self.prompts["iwslt"] = ("Translate '{0}' to french as a Translator.").format("{eng_text}")
        self.prompts["samsum"] = ("Summarise the Dialogue: '{0}' as a Summariser.").format("{dialogue}")  

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

        self.prompts["boolq"] = ("Based on the passage:'{0}'\nAnswer True/False to the question: '{1}'.\nAnswer: {2}.\nExplaination: {3}.\nBased on the passage:'{4}'\nAnswer True/False to the question: '{5}'.\nAnswer: {6}.\nExplaination: {7}.\nBased on the passage:'{8}'\nAnswer True/False to the question: '{9}'.\nAnswer: {10}.\nExplaination: {11}.\nBased on the passage:'{12}'\nAnswer True/False to the question: '{13}'").format( passage1, question1, ans1, exp1, passage2, question2, ans2, exp2,  passage3, question3, ans3, exp3, "{passage}", "{question}" )
        self.prompts["csqa"] = ("Choose the answer.\n{0}\nA bank\nB library\nC department store\nD mall\nE new york\nAnswer: A\nExplanation: {1}.\nChoose the answer.\n{2}\nA doctor\nB bookstore\nC market\nD train station\nE mortuary\nAnswer: B\nExplanation: {3}.\nChoose the answer.\n{4}\nA pants\nB record albums\nC record store\nD cheese\nE wallpaper\nAnswer: E\nExplanation: {5}.\nChoose the answer.\n{6}\nA {7}\nB {8}\nC {9}\nD {10}\nE {11}").format( question_1, exp_1, question_2, exp_2, question_3, exp_3, "{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{text5}" )
        self.prompts["iwslt"] = ("Translate '{0}' to French.\nFrench: {1}.\nTranslate '{2}' to French.\nFrench: {3}.\nTranslate '{4}' to French.\nFrench: {5}.\nTranslate '{6}' to French.").format( ex_en1, ex_fr1, ex_en2, ex_fr2, ex_en3, ex_fr3, "{eng_text}" )
        self.prompts["summarise"] = ("Summarise the Dialogue: '{0}'.\nSummary: {1}.\nSummarise the Dialogue: '{2}'.\nSummary: {3}.\nSummarise the Dialogue: '{4}'.\nSummary: {5}.\nSummarise the Dialogue: '{6}'").format( ex_dialogue1, ex_sum1, ex_dialogue2, ex_sum2, ex_dialogue3, ex_sum3, "{dialogue}" )

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

    def get_prompt(self, task):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        else :
            return self.prompts[task]
        
class generatedknowledge(Promptloader):
    def __init__(self):
        super().__init__()
        self.prompts["boolq"] = ("Based on the passage:'{0}'\nAnswer True/False to the question: '{1}' using interpretation of the passage:{2}.").format("{passage}", "{question}", "{pred}")
        self.prompts["csqa"] = ("Choose the answer.\n{0}\nA {1}\nB {2}\nC {3}\nD {4}\nE {5} using interpretation of the question:{6}").format("{question}", "{text1}", "{text2}", "{text3}", "{text4}", "{text5}", "{pred}")
        self.prompts["iwslt"] = ("Translate '{0}' to french using interpretation of the text:{1}").format("{eng_text}", "{pred}")
        self.prompts["samsum"] = ("Summarise the Dialogue: '{0}' using interpretation of the dialogue:{1}").format("{dialogue}", "{pred}")
 

    def get_prompt(self, task):
        if task not in self.prompts:
            return f"Prompt for '{task}' not found"
        else :
            return self.prompts[task], self.generate_knowledge_prompts[task]