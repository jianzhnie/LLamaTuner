python chatllms/evaluation/evaluate_zh.py \
    --model_name_or_path ~/checkpoints/baichuan7b \
    --split test  \
    --data_path ~/prompt_data/ceval-exam \
    --output_dir ./work_dir/ceval_output
