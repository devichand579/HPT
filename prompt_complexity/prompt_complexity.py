from .prompts import Roleprompt, ZeroshotCoT, threeshotCoT, Leasttomost, generatedknowledge, Promptloader
import argparse
import json
import logging
import re

prompts = {
    1 : Roleprompt(),
    2 : ZeroshotCoT(),
    3 : threeshotCoT(),
    4 : Leasttomost(),
    5 : generatedknowledge()
}

hp_scores = {
    "boolq": 1.71,
    "csqa": 2.52,
    "iwslt": 1.92,
    "samsum": 2.23,
    "humaneval":4.68,
    "gsm8k": 2.14,
    "mmlu": 3.03
}



def main():
