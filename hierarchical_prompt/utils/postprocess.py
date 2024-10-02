from abc import ABC
import logging
import re

class AnswerProcessor(ABC):
    """
    An utility class to post-process the model output for different datasets.
    """
    def __init__(self,name):
        self.dataset_processors = {
            "boolq": self.pp_boolq,
            "csqa": self.pp_csqa,
            "iwslt": self.pp_iwlst,
            "samsum": self.pp_samsum,
            "gsm8k": self.pp_gsm8k,
            "humaneval": self.pp_humaneval,
            "mmlu": self.pp_mmlu
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
                if answer_sentence.startswith('a') or answer_sentence.startswith('a.') or answer_sentence.startswith('a)') or answer_sentence.startswith('( a') or answer_sentence.startswith('** a') or answer_sentence.startswith('**a.') or answer_sentence.startswith('**( a') or answer_sentence.startswith('** a)') or answer_sentence.startswith('a:') or answer_sentence.startswith('**a') :
                    return 0
                elif answer_sentence.startswith('b') or answer_sentence.startswith('b.') or answer_sentence.startswith('b)') or answer_sentence.startswith('( b') or answer_sentence.startswith('** b') or answer_sentence.startswith('**b.') or answer_sentence.startswith('**( b') or answer_sentence.startswith('** b)') or answer_sentence.startswith('b:') or answer_sentence.startswith('**b'):
                    return 1
                elif answer_sentence.startswith('c') or answer_sentence.startswith('c.') or answer_sentence.startswith('c)') or answer_sentence.startswith('( c') or answer_sentence.startswith('** c') or answer_sentence.startswith('**c.') or answer_sentence.startswith('**( c') or answer_sentence.startswith('** c)') or answer_sentence.startswith('c:') or answer_sentence.startswith('**c') :
                    return 2
                elif answer_sentence.startswith('d') or answer_sentence.startswith('d.') or answer_sentence.startswith('d)') or answer_sentence.startswith('( d') or answer_sentence.startswith('** d') or answer_sentence.startswith('**d.') or answer_sentence.startswith('**( d') or answer_sentence.startswith('** d)') or answer_sentence.startswith('d:') or answer_sentence.startswith('**d') :
                    return 3
                elif answer_sentence.startswith('e') or answer_sentence.startswith('e.') or answer_sentence.startswith('e)') or answer_sentence.startswith('( e') or answer_sentence.startswith('** e') or answer_sentence.startswith('**e.') or answer_sentence.startswith('**( e') or answer_sentence.startswith('** e)') or answer_sentence.startswith('e:') or answer_sentence.startswith('**e') :
                    return 4
                else:
                    return 0

    def pp_iwlst(self, passage):
        """Process International Workshop on Spoken Language Translation (IWSLT) text."""
        passage = passage.replace("french:\n\n", "French:").replace("french:\n\n ", "French:")
        lines = passage.split('\n')
        for i, line in enumerate(lines):
            if 'French:' in line:
                french_sentence = lines[i].replace('French:', '').strip()
                return french_sentence

    def pp_samsum(self, passage):
        """Process SAMSum text."""
        result = passage.split("Summary:")[-1].strip()
        return result
    
    def pp_gsm8k(self, text):
        """Process GSM8K text to extract the answer."""
        text = text.lower()
        lines = text.split('\n')
        for line in lines:
            if ('####' in line or 'answer:####' in line or 'answer: ###' in line) and 'after: ####' not in line:
                match = re.search(r'\.?\$?(\d+(?:[,.]\d+)*(?:[,.]\d+)?|\d+([,.]\d+)?)\s*\$?\.?', line)
                if match:
                  return match.group(1) 
       
            if 'answer:' in line:
              match = re.search(r'\*\*.*?(\d+).*?\*\*', line)
              if match:
                return match.group(1) 
        return 0 
            

    def pp_humaneval(self, text):
    # Step 1: Extract the text after 'Code:'
      code_text = text.split("Code:", 1)[-1].strip()  # Extract and strip extra spaces

      # Step 2: Extract the code inside triple backticks 
      code_block = re.search(r'```⁠([\s\S]+?)```', code_text)
      
      if code_block:
          extracted_code = code_block.group(1)
      else:
          extracted_code = code_text  # If no code block is found, return an empty string

      # Step 3: Check if the code starts with "Python" and remove it if present
      if extracted_code.strip().lower().startswith("python"):
          # Remove the word 'Python' from the start
          extracted_code = extracted_code[len("Python"):].strip()
      print(f'*****result = {extracted_code}')
      return [extracted_code]


    
    def pp_mmlu(self, text):
        """Process MMLU text."""
        text = text.lower()

        lines = text.split('\n')
        for i, line in enumerate(lines):
            if "answer:" in line:
                answer_sentence = lines[i].replace('answer:', '').replace('answer is:', '').replace('answer is', '').replace('the correct', '').strip()
                if answer_sentence.startswith('a') or answer_sentence.startswith('a.') or answer_sentence.startswith('a)') or answer_sentence.startswith('( a') or answer_sentence.startswith('** a') or answer_sentence.startswith('**a.') or answer_sentence.startswith('**( a') or answer_sentence.startswith('** a)') or answer_sentence.startswith('a:') or answer_sentence.startswith('**a') :
                    return 0
                elif answer_sentence.startswith('b') or answer_sentence.startswith('b.') or answer_sentence.startswith('b)') or answer_sentence.startswith('( b') or answer_sentence.startswith('** b') or answer_sentence.startswith('**b.') or answer_sentence.startswith('**( b') or answer_sentence.startswith('** b)') or answer_sentence.startswith('b:') or answer_sentence.startswith('**b'):
                    return 1
                elif answer_sentence.startswith('c') or answer_sentence.startswith('c.') or answer_sentence.startswith('c)') or answer_sentence.startswith('( c') or answer_sentence.startswith('** c') or answer_sentence.startswith('**c.') or answer_sentence.startswith('**( c') or answer_sentence.startswith('** c)') or answer_sentence.startswith('c:') or answer_sentence.startswith('**c') :
                    return 2
                elif answer_sentence.startswith('d') or answer_sentence.startswith('d.') or answer_sentence.startswith('d)') or answer_sentence.startswith('( d') or answer_sentence.startswith('** d') or answer_sentence.startswith('**d.') or answer_sentence.startswith('**( d') or answer_sentence.startswith('** d)') or answer_sentence.startswith('d:')or answer_sentence.startswith('**d') :
                    return 3
                else:
                    return 0

            
class AdaptiveProcessor(ABC):
    """
    An utility class to post-process the model output for adaptive hierarchical prompt framework.
    """

    def __init__(self):
        self.processor = self.adaptive_processor
    
    def adaptive_processor(self, text):
        """ Process text for adaptive hierarchical prompt framework."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'Level:' in line:
                ans = lines[i].replace('Level:', '').strip()
                return ans
