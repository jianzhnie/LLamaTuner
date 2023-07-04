CUDA_VISIBLE_DEVICES=0 python cli_demo.py \
    --model_name_or_path ~/checkpoints/baichuan7b \
    --checkpoint_dir ./work_dir/oasst1-baichuan-7b/checkpoint-1100 \
    --trust_remote_code  \
    --double_quant \
    --quant_type nf4 \
    --fp16 \
    --bits 4
