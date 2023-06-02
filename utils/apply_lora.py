"""
Apply the LoRA weights on top of a base model.

Usage:
python3 -m fastchat.model.apply_lora --base ~/model_weights/llama-7b --target ~/model_weights/baize-7b --lora project-baize/baize-lora-7B

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
    load_8bit: bool = False,
    target_model_path: str = None,
    save_target_model: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Applies the LoRA adapter to a base model and saves the resulting target model (optional).

    Args:
        base_model_path (str): The path to the base model to which the LoRA adapter will be applied.
        lora_path (str): The path to the LoRA adapter.
        target_model_path (str): The path where the target model will be saved (if `save_target_model=True`).
        save_target_model (bool, optional): Whether to save the target model or not. Defaults to False.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the target model and its tokenizer.

    """
    # Load the base model and tokenizer
    print(f'Loading the base model from {base_model_path}')
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map='auto',
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
        torch_dtype=torch.float16,
    )

    if save_target_model and target_model_path is not None:
        print(f'Saving the target model to {target_model_path}')
        model.save_pretrained(target_model_path)
        base_tokenizer.save_pretrained(target_model_path)

    return model, base_tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', type=str, required=True)
    parser.add_argument('--target-model-path', type=str, required=True)
    parser.add_argument('--lora-path', type=str, required=True)
    parser.add_argument('--save-target-model', type=bool, default=False)

    args = parser.parse_args()

    apply_lora(args.base_model_path, args.target_model_path, args.lora_path,
               args.save_target_model)
