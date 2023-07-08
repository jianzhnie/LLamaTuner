#!/bin/bash
# nohup sh scripts/finetune/finetune_baichuan_7b_olcc.sh > run2.log 2>&1 &
# nohup sh scripts/multiturn/full-finetune_alpaca_ds.sh  > run2.log 2>&1 &
nohup sh scripts/qlora_finetune/multiturn_llama_finetune.sh > run_generated_chat_vicuna_llama_1gpu.log 2>&1 &
nohup sh scripts/qlora_finetune/multiturn_baichuan_finetune.sh > run_generated_chat_vicuna_baichuan_1gpu.log 2>&1 &
