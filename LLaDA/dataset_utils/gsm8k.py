import re
import os
import json
import random

# --- GSM8K 
GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

def gsm8k_doc_to_text(doc):
    question = doc["question"]
    context = GSM_SYSTEM_PROMPT + '\n\n' + question
    return [{"role": "user", "content": context}], doc['answer']

def gsm8k_extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)")
    match = re.search(r"\\boxed\{(.*)\}", s)
    if match:
        match_str = match.group(1).replace(",", "").replace("+", "").strip()
        match_iter = list(_PAT_LAST_DIGIT.finditer(match_str))
        if match_iter:
            try: return float(match_iter[-1].group())
            except ValueError: return None
    match_iter = list(_PAT_LAST_DIGIT.finditer(s))
    if match_iter:
        try: return float(match_iter[-1].group().replace(",", "").replace("+", "").strip())
        except ValueError: return None
    return None

def gsm8k_is_correct(pred, gt):
    gt_ans = gsm8k_extract_answer(gt)
    if pred is None or gt_ans is None: return False
    return abs(pred - gt_ans) < 1e-5