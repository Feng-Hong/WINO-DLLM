import requests
from openai import AzureOpenAI, OpenAI
import os
from loguru import logger as eval_logger
import time

def sqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if lmms_eval_specific_kwargs["format"] == "default":
        if context:
            context = f"Context: {context}\n"

        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        return f"{pre_prompt}{context}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["format"] == "qwen_vl":
        prompt = "Context: {}\nQuestion: {}\nOptions: {}\nAnswer:"
        context = context if context else "N/A"
        prompt = prompt.format(context, question, choices_str)
        return prompt
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs}")


def sqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def sqa_doc_to_target(doc):
    len_choices = len(doc["choices"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["answer"]]


def sqa_process_results(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = sqa_doc_to_target(doc).strip().lower()
    pred = results[0].strip().lower()
    if pred == target:
        return {"exact_match": 1.0}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0] == target else 0.0
        return {"exact_match": result}
    return {"exact_match": 0.0}


NUM_SECONDS_TO_SLEEP = 5
API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

JUDGE_RULES = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
0 or 1"""

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAI(api_key=API_KEY, base_url=API_URL.rstrip("chat/completions"))
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    client = AzureOpenAI(azure_endpoint=API_URL, api_version="2023-07-01-preview", api_key=API_KEY)


def get_chat_response(content: str, max_tokens: int, retries: int = 5):
    global MODEL_VERSION
    global client

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the correctness of the answer.",
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": MODEL_VERSION,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            return content
        except requests.exceptions.RequestException as e:
            eval_logger.warning(f"Request failed on attempt {attempt+1}: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)
            if attempt == retries - 1:
                eval_logger.error(f"Failed to get response after {retries} attempts")
                return ""
        except Exception as e:
            eval_logger.error(f"Error on attempt {attempt+1}: {e}")
            return ""

def sqa_reasoning_process_results(doc, results):
    scores = []
    for pred in results:
        context, question, choices = doc["hint"], doc["question"], doc["choices"]
        len_choices = len(choices)
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        if context:
            context = f"Context: {context}\n"
        formatted_question = f"{context}{question}\n{choices_str}"
        answer = f'{options[int(doc["answer"])]}. {choices[int(doc["answer"])]}'

        llm_judge_prompt = JUDGE_RULES.format(question=formatted_question, answer=answer, pred=pred)
        llm_judge_score = get_chat_response(llm_judge_prompt, max_tokens=20, retries=3)
        scores.append(llm_judge_score)

    sqa_judge_acc = {"score": scores}
    return {"sqa_judge_acc": sqa_judge_acc}

def sqa_aggregate_judge_results(results):
    total_score = 0
    for result in results:
        try:
            item_score = 1 if "1" in result["score"] or "[1]" in result["score"] else 0
            total_score += item_score
        except:
            eval_logger.warning(f"Failed to convert score to int, getting result: {result['score']}")
            total_score += 0
    return total_score / len(results)