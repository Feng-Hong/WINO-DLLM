# WINO-DLLM

Official implementation for [**WINO+**](https://arxiv.org/pdf/2605.16941), a journal extension of [WINO](https://github.com/Feng-Hong/WINO-DLLM/tree/main).

This repository provides the WINO+ workflow for LLaDA and MMaDA:

- collect offline WINO trajectories for training;
- train WINO+ LoRA adapters;
- merge LoRA adapters into base models;
- evaluate merged WINO+ models.

## WINO+ Trajectory Data Preparation

WINO+ post-training uses offline WINO trajectories. The lightweight data preparation code is under
[`prepare_trainingdata/`](./prepare_trainingdata/). It currently supports GSM8K, Countdown, and IconQA.
Trajectory collection uses the original WINO Python environment; refer to the
[`Feng-Hong/WINO-DLLM` main branch](https://github.com/Feng-Hong/WINO-DLLM/tree/main) for environment setup.

All trajectory collectors write JSONL records with training-facing fields such as:

```json
{
  "prompt_ids": [1, 2, 3],
  "generated_ids": [4, 5, 6],
  "trajectory_accepted": [0, 0, 1],
  "trajectory_proposed": [0, 0, 1],
  "correct": true
}
```

Example: collect LLaDA GSM8K trajectories.

```bash
python -m prepare_trainingdata.llada.prepare_gsm8k \
  --model-path /path/to/LLaDA-8B-Instruct \
  --output-file ./data/gsm8k_processed.jsonl

python -m prepare_trainingdata.llada.collect_gsm8k_trajectories \
  --model-path /path/to/LLaDA-8B-Instruct \
  --input-file ./data/gsm8k_processed.jsonl \
  --output-file ./data/gsm8k_wino_trajectory.jsonl
```

Example: collect MMaDA IconQA trajectories.

```bash
python -m prepare_trainingdata.mmada.prepare_iconqa \
  --model-path /path/to/MMaDA-8B-MixCoT \
  --input-file /path/to/iconqa_train_dataset.jsonl \
  --image-root /path/to/iconqa/images \
  --output-file ./data/iconqa_processed.jsonl

python -m prepare_trainingdata.mmada.collect_iconqa_trajectories \
  --mmada-model-path /path/to/MMaDA-8B-MixCoT \
  --vq-model-path showlab/magvitv2 \
  --input-file ./data/iconqa_processed.jsonl \
  --image-root /path/to/iconqa/images \
  --output-file ./data/iconqa_wino_trajectory.jsonl
```

Filter correct trajectories before training:

```bash
python -m prepare_trainingdata.common.filter_trajectories \
  --input-file ./data/iconqa_wino_trajectory.jsonl \
  --output-file ./data/iconqa_wino_trajectory_filtered.jsonl
```

See [`prepare_trainingdata/README.md`](./prepare_trainingdata/README.md) for task-specific details.

## WINO+ LoRA Training

WINO+ training uses separate `uv` environments from the LLaDA and MMaDA evaluation environments above. Do not reuse
or modify existing external project environments when setting up these training runs.

Create a dedicated LLaDA training environment:

```bash
cd /path/to/WINO-DLLM
uv venv --python 3.10 training/llada/.venv
source training/llada/.venv/bin/activate
uv pip install -r training/llada/requirements.txt
deactivate
```

Create a dedicated MMaDA training environment:

```bash
cd /path/to/WINO-DLLM
uv venv --python 3.11 training/mmada/.venv
source training/mmada/.venv/bin/activate
uv pip install -r training/mmada/requirements.txt
deactivate
```

Activate only the matching training environment before launching each trainer.

### LLaDA WINO+ Training

The LLaDA trainer supports the two-stage setup used in the paper: first train on GSM8K trajectories, then continue
from the first adapter on Countdown trajectories.

Edit [`training/llada/config/llada_wino_plus_two_stage.yaml`](./training/llada/config/llada_wino_plus_two_stage.yaml)
to set the base model path, trajectory files, and output directories, then run:

```bash
source training/llada/.venv/bin/activate
python -m training.llada.train_wino_plus_lora \
  --config training/llada/config/llada_wino_plus_two_stage.yaml
```

### MMaDA WINO+ Training

The MMaDA trainer consumes tokenized trajectory JSONL files whose `prompt_ids` already contain image tokens and the
text prompt. It does not reload the VQ model during training.

For 8 GPU DeepSpeed ZeRO-3 training:

```bash
source training/mmada/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  --config_file training/mmada/accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml \
  -m training.mmada.train_wino_plus_lora \
  --config training/mmada/config/mmada_wino_plus_lora.yaml
```

You can override config values from the command line:

```bash
source training/mmada/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  --config_file training/mmada/accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml \
  -m training.mmada.train_wino_plus_lora \
  --config training/mmada/config/mmada_wino_plus_lora.yaml \
  model.mmada.tokenizer_path=/path/to/LLaDA-8B-Instruct \
  model.mmada.pretrained_model_path=/path/to/MMaDA-8B-MixCoT \
  dataset.params.train_trajectory_path=/path/to/iconqa_wino_trajectory_filtered.jsonl
```

For a short smoke test on real trajectory data, add:

```bash
training.max_train_steps=1 \
experiment.output_dir=/tmp/mmada_wino_plus_smoke
```

## Merge LoRA Adapters

After WINO+ LoRA training, merge a single adapter into the base model for evaluation.

Merge LLaDA LoRA:

```bash
source training/llada/.venv/bin/activate
python -m training.llada.merge_lora \
  --base-model /path/to/LLaDA-8B-Instruct \
  --adapter /path/to/llada/final_adapter_or_checkpoint \
  --output-dir /path/to/merged-llada-winoplus
```

Merge MMaDA LoRA:

```bash
source training/mmada/.venv/bin/activate
python -m training.mmada.merge_lora \
  --base-model /path/to/MMaDA-8B-MixCoT \
  --adapter /path/to/mmada/adapter \
  --output-dir /path/to/merged-mmada-winoplus
```

## Evaluation of WINO+ Models

For LLaDA, set the evaluation config `model_path` to the merged WINO+ model and use:

LLaDA configs support:

```yaml
method: confidence_threshold
```

For MMaDA, evaluate a merged WINO+ model with:

```bash
cd MMaDA
MODEL_PATH=/path/to/merged-mmada-winoplus \
NGPU=8 \
bash scripts/eval_winoplus.sh
```
