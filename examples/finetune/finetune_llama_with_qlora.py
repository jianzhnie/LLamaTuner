from typing import Dict

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          LlamaTokenizer, Trainer, TrainingArguments)

DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'


def print_trainable_parameters(model: AutoModelForCausalLM) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: \
            {100 * trainable_params / all_param}')


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


if __name__ == '__main__':
    model_id = 'decapoda-research/llama-7b-hf'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    """
    - load_in_4bit: The model will be loaded in the memory with 4-bit precision.
    - bnb_4bit_use_double_quant: We will do the double quantization proposed by QLoRa.
    - bnb_4bit_quant_type: This is the type of quantization. “nf4” stands for 4-bit NormalFloat.
    - bnb_4bit_compute_dtype: While we load and store the model in 4-bit,
        we will partially dequantize it when needed and do all the computations with a 16-bit precision (bfloat16).
    """
    # So now we can load the model in 4-bit:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={'': 0})

    # Then, we enable gradient checkpointing, to reduce the memory footprint of the model:
    model.gradient_checkpointing_enable()
    # Then, we load the tokenizer:
    if model.config.model_type == 'llama':
        # Due to the name of Transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(
            model_id,
            padding_side='right',
            use_fast=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side='right',
            use_fast=True,
        )
    # Preprocessing the GPT model for LoRa
    model = prepare_model_for_kbit_training(model)
    # This is where we use PEFT. We prepare the model for LoRa, adding trainable adapters for each layer.
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    # We can now add the adapters to the model:
    model = get_peft_model(model, config)
    # We can now print the number of trainable parameters in the model:
    print_trainable_parameters(model)

    # Get your dataset ready
    # For this demo, I use the “english_quotes” dataset. This is a dataset made of famous quotes distributed under a CC BY 4.0 license.
    data = load_dataset('Abirate/english_quotes')
    data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)

    # Add special tokens to tokenizer if they are not already present
    special_tokens_dict: Dict[str, str] = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    trainer = Trainer(
        model=model,
        train_dataset=data['train'],
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_steps=2,
            max_steps=1000,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir='outputs',
            optim='paged_adamw_8bit',
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
