import re
import os
import json
import random
ARC_SYSTEM_PROMPT = """You will be given a multiple choice question and a list of options. You need to first reason step by step, and then select the correct option (A, B, C, D). Wrap the single letter of the correct option in \\boxed{...}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>
"""

def arc_doc_to_text(doc):
    question = doc['question']
    options_text, options_label = doc['choices']['text'], doc['choices']['label']
    correct_option_letter = doc['answerKey']
    options_str = [f"{label.strip()}. {text.strip()}\n" for label, text in zip(options_label, options_text)]
    full_question = f"{question.strip()}\n" + ''.join(options_str)
    context = ARC_SYSTEM_PROMPT + '\n\n' + full_question.strip()
    return [{"role": "user", "content": context}], correct_option_letter

def arc_extract_answer(s):
    answer = s.split('<answer>')[-1].split('</answer>')[0].strip()
    answer = answer.split('\\boxed{')[-1].split('}')[0].strip()
    answer = answer.split('\n')[-1].strip().split('.')[0]
    return answer.upper() if answer.isalpha() else None

def arc_is_correct(pred, gt):
    if pred is None or gt is None: return False
    return pred == gt
