import re
import os
import json
import random
import textwrap
def humaneval_doc_to_text(doc):
    prompt = f"Write a solution to the following problem and make sure that it passes the tests:\n```python\n{doc['prompt']}\n```"
    context = [{"role": "user", "content": prompt}]
    trailing_prompt = f"Here is the completed function:\n```python\n{doc['prompt']}\n"
    return context, doc, trailing_prompt

def humaneval_extract_answer(generated_text, doc):
    text_with_prefix = f"```python\n{doc['prompt']}\n" + generated_text
    entry_point = doc['entry_point']
    pattern = re.compile(rf"def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL)
    match = pattern.search(text_with_prefix)
    if match: return match.group(1).rstrip()
    fallback = re.split(r"```", generated_text, maxsplit=1)
    if fallback: return fallback[0].rstrip()
    return generated_text
