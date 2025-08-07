import re
import os
import json
import random
import pandas as pd
import datasets
SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Example:
Puzzle: 3102200002100320
Answer: 3142243142131324

Puzzle: 0140432014020000
Answer: 2143432114323214

Puzzle: 4310010000303420
Answer: 4312214312343421

Puzzle: 2004001203201003
Answer: 2134341243211243

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

def sudoku_read_data(data_path):
    df = pd.read_csv(data_path, dtype={'Puzzle': str, 'Solution': str})
    return datasets.Dataset.from_pandas(df)

def sudoku_doc_to_text(doc):
    question = f"Solve the following Sudoku puzzle: {doc['Puzzle']}\n"
    context = SUDOKU_SYSTEM_PROMPT + '\n\n' + question
    return [{"role": "user", "content": context}, ], context
    
def sudoku_evaluate_and_extract(out_str, target, puzzle):
    assert len(puzzle) == 16, f"Puzzle length is {len(puzzle)}, expected 16"
    
    empty_indices = [i for i in range(16) if puzzle[i] == "0"]
    empty_cells = len(empty_indices)
    
    solution_str = ""
    patterns = [
        r"<answer>.*?```\s*([\d\s]+)```",
        r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
        r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
        r".*?(\d{16})\s*</answer>",
        r"\b(\d{16})\b",
    ]

    for pattern in patterns:
        if solution_str:
            break
        match = re.search(pattern, out_str, re.DOTALL)
        if match and match.group(1).strip():
            solution_str = match.group(1).strip()

    solution_str = re.sub(r"\s", "", solution_str)

    # Handle solution length
    if not solution_str:
        correct_cells = 0
    else:
        if len(solution_str) < 16:
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            solution_str = solution_str[:16]
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == target[i])
    
    return correct_cells, empty_cells, solution_str

