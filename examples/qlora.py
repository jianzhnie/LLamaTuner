import transformers
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer, Trainer)

DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )


if __name__ == '__main__':
    model_id = 'decapoda-research/llama-7b-hf'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={'': 0})

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

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(r=8,
                        lora_alpha=32,
                        target_modules=['q_proj', 'v_proj'],
                        lora_dropout=0.05,
                        bias='none',
                        task_type='CAUSAL_LM')

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    data = load_dataset('Abirate/english_quotes')
    data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)

    # Add special tokens to tokenizer if they are not already present
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)

    trainer = Trainer(
        model=model,
        train_dataset=data['train'],
        args=transformers.TrainingArguments(per_device_train_batch_size=1,
                                            gradient_accumulation_steps=4,
                                            warmup_steps=2,
                                            max_steps=10,
                                            learning_rate=2e-4,
                                            fp16=True,
                                            logging_steps=1,
                                            output_dir='outputs',
                                            optim='paged_adamw_8bit'),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,
                                                                   mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
