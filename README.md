This repository provides scripts and instructions to evaluate [WINO](https://arxiv.org/pdf/2507.18578) on LLaDA and MMaDA.


## Evaluation of WINO on LLaDA

1. Installation
We recommend using [uv](https://github.com/astral-sh/uv) for dependency and virtual environment management.
```bash
pipx install uv # or pip install uv
cd LLaDA
uv venv --python 3.11 dev
source dev/bin/activate
uv pip install -r requirements.txt
```

2. Prepare Model and Datasets

Before running inference or evaluation, please download the following models and datasets from [Hugging Face](https://huggingface.co/) into the specified local directories (e.g., [`./LLaDA/models/`](./LLaDA/models/) and [`./LLaDA/data/`](./LLaDA/data/)). 

You may use either `huggingface-cli` or the Python `datasets` library to complete the download.

| Model Name         | Hugging Face Repo                                               | Local Path                     |
|--------------------|------------------------------------------------------------------|--------------------------------|
| LLaDA-8B-Instruct  | [`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | `./LLaDA/models/LLaDA-8B-Instruct/`  |

| Dataset Name  | Hugging Face Repo                                                                 | Local Path          |
|---------------|------------------------------------------------------------------------------------|---------------------|
| GSM8K         | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)                    | `./LLaDA/data/gsm8k/`     |
| MATH-500      | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `./LLaDA/data/math500/`   |
| HumanEval     | [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) | `./LLaDA/data/humaneval/` |
| ARC (AI2)     | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc)              | `./LLaDA/data/ai2_arc/`       |

Datasets not listed above are already included in the [`./LLaDA/data/`](./LLaDA/data/) directory

3. Quick Demo

Please make sure to set the correct model path in generate.py.

```bash
python generate.py
```
4. Evaluation

To evaluate WINO on a benchmark such as GSM8K. Please configure the model and data paths in the corresponding config file.

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config ./configs/gsm8k.yaml
```
All available config files can be found in the [`./LLaDA/configs/`](./LLaDA/configs/) directory.



## Evaluation of WINO on MMaDA

We evaluate **WINO** using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

To run the evaluation, follow these steps:

1. **Install MMaDA dependencies**
```bash
cd MMaDA
# pipx install uv
uv venv --python 3.11 dev
source dev/bin/activate
uv pip install -r requirements.txt
```

A quick inference demo can be performed after this step.
```bash
python generate_demo.py
```

2. **Install lmms-eval dependencies**
```bash
cd lmms_eval
uv pip install -e .
```

3. **Set some necessary environmental variables**
   Some environmental variables are necessary for certain tasks to run.
```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
```

Once all dependencies are installed and your API key is set, you can run the evaluation script directly:

```bash
cd ..
# Evaluating MMaDA on the reported six multimodel benchmarks
bash scripts/eval_baseline.sh
# Evaluating WINO on the reported six multimodel benchmarks
bash scripts/eval_wino.sh
```

