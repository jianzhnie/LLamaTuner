"""
Apply the LoRA weights on top of a base model.

Usage:
python3 apply_lora.py --base_model_path ~/model_weights/llama-7b --target_model_path ~/model_weights/baize-7b \
    --lora_path project-baize/baize-lora-7B

Dependency:
pip3 install git+https://github.com/huggingface/peft.git@2822398fbe896f25d4dac5e468624dc5fd65a51b
"""
import argparse
from typing import Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_lora(
    base_model_path: str,
    lora_model_path: str,
    target_model_path: str = None,
    cache_dir: str = None,
    use_auth_token: str = True,
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Applies the LoRA adapter to a base model and saves the resulting target model (optional).

    Args:
        base_model_path (str): The path to the base model to which the LoRA adapter will be applied.
        lora_model_path (str): The path to the LoRA adapter.
        target_model_path (str): The path where the target model will be saved (if `save_target_model=True`).
        cache_dir (str): The path to the cache directory.
        use_auth_token (bool): Whether to use an authentication token when downloading the model.
        trust_remote_code (bool): Whether to trust remote code when downloading the model.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the target model and its tokenizer.

    """
    # Load the base model and tokenizer
    print(f'Loading the base model from {base_model_path}')
    # Set configuration kwargs for tokenizer.
    config_kwargs = {
        'cache_dir': cache_dir,
        'use_auth_token': use_auth_token,
        'trust_remote_code': trust_remote_code,
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **config_kwargs,
    )

    # Load the tokenizer
    print(f'Loading the tokenizer from {base_model_path}')
    # Due to the name of Transformers' LlamaTokenizer, we have to do this
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        **config_kwargs,
    )

    # Load the LoRA adapter
    print(f'Loading the LoRA adapter from {lora_model_path}')
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    print('Applying the LoRA')
    model = model.merge_and_unload()

    if target_model_path is not None:
        print(f'Saving the target model to {target_model_path}')
        model.save_pretrained(target_model_path)
        tokenizer.save_pretrained(target_model_path)

    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', type=str, required=True)
    parser.add_argument('--target-model-path', type=str, default=None)
    parser.add_argument('--lora-model-path', type=str, required=True)
    args = parser.parse_args()

    apply_lora(base_model_path=args.base_model_path,
               lora_path=args.lora_model_path,
               target_model_path=args.target_model_path)
