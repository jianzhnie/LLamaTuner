python llamatuner/train/sft/train_lora.py \
    --model_name_or_path facebook/opt-125m \
    --dataset alpaca \
    --output_dir work_dir/lora-finetune \
    --wandb_project llamatuner \
    --wandb_run_name alpaca_opt-125m_lora-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --optim "adamw_torch" \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --trust_remote_code \
    --model_max_length 128 \
    --do_train \
    --do_eval
