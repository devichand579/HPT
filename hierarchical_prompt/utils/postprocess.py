from abc import ABC
import logging

class AnswerProcessor(ABC):
    """
    An utility class to post-process the model output for different datasets.
    """
    def __init__(self,name):
        self.dataset_processors = {
            "boolq": self.pp_boolq,
            "csqa": self.pp_csqa,
            "iwslt": self.pp_iwlst,
            "samsum": self.pp_samsum
        }
        self.processor = self.dataset_processors.get(name, lambda x: x)
        logging.info(f"***{name} post-processor created successfully***")

    def pp_boolq(self, text):
        """Process Boolean Questions (BoolQ) text."""
        text = text.lower()
        lines = text.split('\n')
        l1 = ['yes', 'true']
        l2 = ['no', 'false']
        for line in lines:
            if "answer:" in line:
                answer_sentence = line.replace('answer:', '').strip()
                print(answer_sentence)
                for word in l1:
                    if word in answer_sentence:
                        return 1
                for word in l2:
                    if word in answer_sentence:
                        return 0
                return 0

    def pp_csqa(self, text):
        """Process Commonsense QA (CSQA) text."""
        text = text.lower()
        lines = text.split('\n')
        answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

        for i, line in enumerate(lines):
            if "answer:" in line:
                answer_sentence = lines[i].replace('answer:', '').strip()
                print(answer_sentence)
                for char in answer_map:
                    if char in answer_sentence:
                        return answer_map[char]
                return 0

    def pp_iwlst(self, passage):
        """Process International Workshop on Spoken Language Translation (IWSLT) text."""
        lines = passage.split('\n')
        for i, line in enumerate(lines):
            if 'French:' in line:
                french_sentence = lines[i].replace('French:', '').strip()
                return french_sentence

    def pp_samsum(self, passage):
        """Process SAMSum text."""
        result = passage.split("Summary:")[-1].strip()
        return result
