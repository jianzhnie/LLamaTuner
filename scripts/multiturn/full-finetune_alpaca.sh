
<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multiturn.py \
    --model_name_or_path  ~/checkpoints/baichuan7b \
    --data_path ~/prompt_data/sharegpt_clean/sharegpt_clean.json \
    --output_dir work_dir/multiturn_full-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --trust_remote_code \
    --lazy_preprocess True
