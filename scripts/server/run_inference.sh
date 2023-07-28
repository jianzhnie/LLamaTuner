# vicuna_merge
CUDA_VISIBLE_DEVICES=4 python cli_demo.py \
    --model_name_or_path ~/checkpoints/baichuan7b/ \
    --checkpoint_dir ./work_dir/vicuna_merge_vicuna-baichuan-7b-1gpu/checkpoint-12000 \
    --prompt_template vicuna \
    --trust_remote_code \
    --double_quant \
    --quant_type nf4 \
    --fp16 \
    --bits 4

CUDA_VISIBLE_DEVICES=3 python cli_demo.py \
    --model_name_or_path ~/checkpoints/baichuan7b/ \
    --checkpoint_dir ./work_dir/vicuna_merge-llama-7b-1gpu/checkpoint-3200 \
    --prompt_template vicuna \
    --trust_remote_code \
    --double_quant \
    --quant_type nf4 \
    --fp16 \
    --bits 4


# generated_chat_vicuna
CUDA_VISIBLE_DEVICES=3 python cli_demo.py \
    --model_name_or_path ~/checkpoints/baichuan7b/ \
    --checkpoint_dir ./work_dir/generated_chat_vicuna-baichuan-7b-1gpu/checkpoint-10500 \
    --prompt_template vicuna \
    --trust_remote_code \
    --double_quant \
    --quant_type nf4 \
    --fp16 \
    --bits 4


# generated_chat_vicuna
CUDA_VISIBLE_DEVICES=4 python server/single_chat.py \
    --model_name_or_path ./work_dir/vicuna_merge_vicuna-baichuan-7b-1gpu/