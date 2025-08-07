import re
import os
import json
import random
import textwrap
def mbpp_read_data(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        prompt = format_test_example(q, test, code=None)
        entry_point = re.search(r"assert\s+([a-zA-Z0-9_]+)\s*\(", test[0]).group(1)
        yield {
            'task_id': ex['task_id'],
            'prompt': prompt,
            'entry_point': entry_point,
            'examples': examples_str,
        }

def mbpp_extract_answer(text, entry_point):
    pattern = re.compile(
        rf"(def\s+{entry_point}\s*\(.*?\):\n.*?)(?=^```)",
        re.DOTALL | re.MULTILINE
    )
    match = pattern.search(text)
    if match:
        return match.group(1).rstrip()
    
    return textwrap.indent(text, " " * 4)


def mbpp_doc_to_text(doc):
    prompt = 'You are an expert Python programmer. Please write a python function to solve the following problem:\n' + doc['prompt']
    context = [{"role": "user", "content": prompt}]
    return context, doc, f">>> Code:\n```python\n"

