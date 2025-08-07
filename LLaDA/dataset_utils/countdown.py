import re
import os
import json
import random
CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
    + """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except:
        return s
    
def countdown_doc_to_text(doc, use_fewshot = False):
    assert use_fewshot is False, "fewshot is not supported for countdown"
    if use_fewshot:
        context = [
            {"role": "user", "content": CTD_SYSTEM_PROMPT + "\n\nThere are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"},
            {"role": "assistant", "content": "<reasoning>There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n</reasoning>\n<answer>\n\\boxed{6}\n</answer>"},
            {"role": "user", "content": CTD_SYSTEM_PROMPT + "\n\nIf there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"},
            {"role": "assistant", "content": "<reasoning>There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n</reasoning>\n<answer>\n\\boxed{5}\n</answer>"},
            {"role": "user", "content": CTD_SYSTEM_PROMPT + "\n\nLeah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"},
            {"role": "assistant", "content": "<reasoning>Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n</reasoning>\n<answer>\n\\boxed{39}\n</answer>"},
            {"role": "user", "content": CTD_SYSTEM_PROMPT + "\n\nJason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"},
            {"role": "assistant", "content": "<reasoning>Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n</reasoning>\n<answer>\n\\boxed{8}\n</answer>"},
            {"role": "user", "content": CTD_SYSTEM_PROMPT + "\n\n" + doc['question'] + "\n"},
        ]
        return context
    else:
        target = int(doc['output'])
        num_str = doc['input']
        nums = [int(i) for i in num_str.split(',')]
        question = f"Numbers: {nums}\nTarget: {target}"
        context = CTD_SYSTEM_PROMPT + '\n\n' + question
        return [{"role": "user", "content": context}, ], context
def countdown_extract_answer(generated_text):
    equation = ""
    try:
        equation = remove_boxed(last_boxed_only_string(generated_text))
    except:
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1).strip()
        else:
            equation = generated_text
    # Replace LaTeX operators with Python operators
    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

    # Check for equation with equals sign and extract only the expression part
    equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()
    
    return equation

def countdown_is_correct(equation, target, numbers):
    is_valid = _countdown_validate_equation(equation, numbers)
    is_correct = False
    if is_valid:
        result = _countdown_evaluate_equation(equation)
        if target is not None and abs(result - target) < 1e-5:
            is_correct = True
    
    return is_correct

def _countdown_validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        return numbers_in_eq == available_numbers
    except:
        return False

def _countdown_evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation."""
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        result = eval(equation_str.strip(), {"__builtins__": None}, {})
        return result
    except Exception:
        return float("Inf")
      
def countdown_read_data(data_path):
    with open(data_path, 'r') as f:
        return [json.loads(line) for line in f]
