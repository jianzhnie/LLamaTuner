python cli_demo.py \
    --model_name_or_path ~/checkpoints/baichuan7b \
    --checkpoint_dir ./work_dir/olcc-baichuan-7b/checkpoint-700 \
    --trust_remote_code  \
    --double_quant \
    --quant_type nf4 \
    --fp16 \
    --bits 4
