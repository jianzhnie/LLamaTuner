python gradio_qlora_webserver.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path ./work_dir/oasst1-llama-7b/checkpoint-831/adapter_model \
    --quant_type nf4 \
    --double_quant \
    --bits 4 \
    --fp16
