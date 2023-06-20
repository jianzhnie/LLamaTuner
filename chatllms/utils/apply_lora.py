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
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


def apply_lora(
    base_model_path: str,
    lora_path: str,
    target_model_path: str = None,
    load_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Applies the LoRA adapter to a base model and saves the resulting target model (optional).

    Args:
        base_model_path (str): The path to the base model to which the LoRA adapter will be applied.
        lora_path (str): The path to the LoRA adapter.
        target_model_path (str): The path where the target model will be saved (if `save_target_model=True`).
        load_8bit (bool): Whether to load the base model in 8-bit precision.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the target model and its tokenizer.

    """
    # Load the base model and tokenizer
    print(f'Loading the base model from {base_model_path}')
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=load_8bit,
        device_map='auto',
        torch_dtype=torch.float16,
        use_auth_token=True,
        trust_remote_code=True,
    )

    # Load the tokenizer
    if base_model.config.model_type == 'llama':
        # Due to the name of Transformers' LlamaTokenizer, we have to do this
        base_tokenizer = LlamaTokenizer.from_pretrained(
            base_model_path,
            padding_side='right',
            use_fast=True,
        )
    else:
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            padding_side='right',
            use_fast=True,
        )

    # Load the LoRA adapter
    print(f'Loading the LoRA adapter from {lora_path}')
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
    )
    print('Applying the LoRA')
    model = model.merge_and_unload()

    if target_model_path is not None:
        print(f'Saving the target model to {target_model_path}')
        model.save_pretrained(target_model_path)
        base_tokenizer.save_pretrained(target_model_path)

    return model, base_tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', type=str, required=True)
    parser.add_argument('--target-model-path', type=str, default=None)
    parser.add_argument('--lora-path', type=str, required=True)
    parser.add_argument('--load_8bit', type=bool, default=False)

    args = parser.parse_args()

    apply_lora(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        target_model_path=args.target_model_path,
        load_8bit=args.load_8bit,
    )
