"""
    An utility class for evaluating BoolQ, CSQA, IWSLT 2017 and Samsum datasets.
"""
from abc import ABC
import logging

class Eval(ABC):
    """
    compute_cls_accuracy(preds, gts)
        Computes classification accuracy.
    compute_f1_score(preds,gts)
        Computes F1 score
    compute_bleu(preds, gts)
        Computes the BLEU score for translation tasks.
    compute_rouge(preds,gts)
        Computes the Rouge score for summarisation tasks.
 
    """
    def __init__(self,name):
        self.methods = {
            "boolq": [self.compute_cls_accuracy, self.compute_f1_score],
            "csqa": [self.compute_cls_accuracy, self.compute_f1_score],
            "iwslt": [self.compute_bleu],
            "samsum": [self.compute_rouge]
        }
        self.metric = self.methods.get(name, lambda x: x)
        logging.info(f"***{name} evaluator created successfully***")
    
    @staticmethod
    def compute_cls_accuracy(preds, gts):
        """
        Computes classification accuracy based on predictions and ground truths.

        Parameters:
        -----------
        preds : list
            A list of predictions.
        gts : list
            A list of ground truths.

        Returns:
        --------
        float
            The classification accuracy.
        """
        try:
            preds = [str(pred).lower() for pred in preds]
            gts = [str(gt).lower() for gt in gts]
        except AttributeError:
            print("Something in either preds or gts can not be convert to a string.")
            
        if not isinstance(preds, list):
            preds = [preds]
            gts = [gts]

        return sum(a == b for a, b in zip(preds, gts)) / len(preds)
    
    @staticmethod
    def compute_f1_score(preds, gts):
        """
        Computes the F1 score based on predictions and ground truths.

        Parameters:
        -----------
        preds : list
            A list of predictions.
        gts : list
            A list of ground truths.

        Returns:
        --------
        float
            The F1 score.
        """
        try:
            preds = [str(pred).lower() for pred in preds]
            gts = [str(gt).lower() for gt in gts]
        except AttributeError:
            print("Something in either preds or gts cannot be converted to a string.")
            return 0.0
            
        if not isinstance(preds, list):
            preds = [preds]
            gts = [gts]

        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        for pred, gt in zip(preds, gts):
            if pred == gt:
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return f1_score


    @staticmethod
    def compute_bleu(preds, gts):
        """
        Computes the BLEU score for translation tasks.

        Parameters:
        -----------
        preds : list
            A list of predictions.
        gts : list
            A list of ground truth translations.

        Returns:
        --------
        float
            The BLEU score.
        """
        from .bleu.bleu import Bleu
        metric = Bleu()
        results = metric.compute(predictions=preds, references=gts)
        return results['bleu']
    
    @staticmethod
    def compute_rouge(preds, gts):
        """
        Computes the Rouge score for Summarisation tasks.

        Parameters:
        -----------
        preds : list
            A list of predictions.
        gts : list
            A list of ground truth summaries.

        Returns:
        --------
        float
            The set of Rouge scores.
        """
        from .rouge.rouge import Rouge
        metric = Rouge
        results = metric.compute(predictions=preds, references=gts)
        return results


    