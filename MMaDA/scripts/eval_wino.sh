SCRIPT_PATH=$(realpath "$0")
cd "$(dirname "$SCRIPT_PATH")/.."
source dev/bin/activate

GEN_METHOD=wino # can be chosen from [default, wino]
GEN_LENGTH=256
DIFF_STEP=256
BLOCK_LENGTH=128
TH=0.5
TH_BACK=0.9

# Runing on MMMU-val
accelerate launch --num_processes=${NGPU} --main_process_port=12345 -m lmms_eval \
    --model mmada \
    --model_args=pretrained=Gen-Verse/MMaDA-8B-MixCoT,gen_method=${GEN_METHOD},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${TH},threshold_back=${TH_BACK},reasoning=True \
    --tasks mmmu_val_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "./results"

# Runing on MathVista-mini
accelerate launch --num_processes=${NGPU} --main_process_port=12345 -m lmms_eval \
    --model mmada \
    --model_args=pretrained=Gen-Verse/MMaDA-8B-MixCoT,gen_method=${GEN_METHOD},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${TH},threshold_back=${TH_BACK},reasoning=True \
    --tasks mathvista_testmini_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "./results"

# Runing on MathVision
accelerate launch --num_processes=${NGPU} --main_process_port=12345 -m lmms_eval \
    --model mmada \
    --model_args=pretrained=Gen-Verse/MMaDA-8B-MixCoT,gen_method=${GEN_METHOD},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${TH},threshold_back=${TH_BACK},reasoning=True \
    --tasks mathvision_test_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "./results"

# Runing on AI2D
accelerate launch --num_processes=${NGPU} --main_process_port=12345 -m lmms_eval \
    --model mmada \
    --model_args=pretrained=Gen-Verse/MMaDA-8B-MixCoT,gen_method=${GEN_METHOD},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${TH},threshold_back=${TH_BACK},reasoning=True \
    --tasks ai2d_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "./results"

# Runing on ScienceQA-Img
accelerate launch --num_processes=${NGPU} --main_process_port=12345 -m lmms_eval \
    --model mmada \
    --model_args=pretrained=Gen-Verse/MMaDA-8B-MixCoT,gen_method=${GEN_METHOD},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${TH},threshold_back=${TH_BACK},reasoning=True \
    --tasks scienceqa_img_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "./results"

# Runing on Flickr30K
accelerate launch --num_processes=${NGPU} --main_process_port=12345 -m lmms_eval \
    --model mmada \
    --model_args=pretrained=Gen-Verse/MMaDA-8B-MixCoT,gen_method=${GEN_METHOD},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${TH},threshold_back=${TH_BACK},reasoning=False \
    --tasks flickr30k_test_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "./results"