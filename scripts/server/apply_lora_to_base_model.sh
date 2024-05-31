CUDA_VISIBLE_DEVICES=0 python llamatuner/utils/apply_lora.py \
        --base-model-path ~/checkpoints/baichuan7b/ \
        --lora-model-path ./work_dir/vicuna_merge_vicuna-baichuan-7b-1gpu/checkpoint-15000 \
        --target-model-path ./work_dir/vicuna_merge_vicuna-baichuan-7b-1gpu/merged_model
