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
        for i, line in enumerate(lines):
            if "answer:" in line:
                answer_sentence = lines[i].replace('answer:', '').strip()
                print(answer_sentence)
                if answer_sentence.startswith('a') or answer_sentence.startswith('a.') or answer_sentence.startswith('a)') or answer_sentence.startswith('( a') or answer_sentence.startswith('** a') or answer_sentence.startswith('**a.') or answer_sentence.startswith('**( a') or answer_sentence.startswith('** a)'):
                    return 0
                elif answer_sentence.startswith('b') or answer_sentence.startswith('b.') or answer_sentence.startswith('b)') or answer_sentence.startswith('( b') or answer_sentence.startswith('** b') or answer_sentence.startswith('**b.') or answer_sentence.startswith('**( b') or answer_sentence.startswith('** b)'):
                    return 1
                elif answer_sentence.startswith('c') or answer_sentence.startswith('c.') or answer_sentence.startswith('c)') or answer_sentence.startswith('( c') or answer_sentence.startswith('** c') or answer_sentence.startswith('**c.') or answer_sentence.startswith('**( c') or answer_sentence.startswith('** c)'):
                    return 2
                elif answer_sentence.startswith('d') or answer_sentence.startswith('d.') or answer_sentence.startswith('d)') or answer_sentence.startswith('( d') or answer_sentence.startswith('** d') or answer_sentence.startswith('**d.') or answer_sentence.startswith('**( d') or answer_sentence.startswith('** d)'):
                    return 3
                elif answer_sentence.startswith('e') or answer_sentence.startswith('e.') or answer_sentence.startswith('e)') or answer_sentence.startswith('( e') or answer_sentence.startswith('** e') or answer_sentence.startswith('**e.') or answer_sentence.startswith('**( e') or answer_sentence.startswith('** e)'):
                    return 4
                else:
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
