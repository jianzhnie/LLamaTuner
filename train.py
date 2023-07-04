import argparse
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, PreTrainedModel,
                          PreTrainedTokenizer, Trainer)

from chatllms.data.sft_dataset import (AlpacaDataset,
                                       DataCollatorForSupervisedDataset)
from chatllms.utils.model_utils import (add_special_tokens_if_missing,
                                        safe_save_model_for_hf_trainer)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained.'
        })
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Enables using Huggingface auth token from Git Credentials.'
        })


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={'help': 'Path to the training data.'})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    model_max_length: int = field(
        default=512,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


def load_model_tokenizer(args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pre-trained model and tokenizer for natural language processing tasks.

    Args:
        args: An object containing the input arguments.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    # Determine the torch data type based on the input arguments
    torch_dtype = torch.float16 if args.fp16 else (
        torch.bfloat16 if args.bf16 else torch.float32)

    config_kwargs = {
        'cache_dir': args.cache_dir,
        'use_auth_token': args.use_auth_token,
        'trust_remote_code': args.trust_remote_code,
    }

    # Load the pre-trained model
    print(f'Loading Model from {args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        **config_kwargs,
    )

    # Enable model parallelism
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if args.gradient_checkpointing:
        logging.warning('Using gradient checkpointing...')
        model.enable_input_require_grads()
        model.config.use_cache = False  # Turn off when gradient checkpointing is enabled

    # Load the tokenizer
    print(f'Loading tokenizer from {args.model_name_or_path}...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='right',
        model_max_length=args.model_max_length,
        use_fast=False,
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,
        **config_kwargs,
    )

    return model, tokenizer


def train() -> None:
    """
    Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.

    Returns:
        None

    """
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    (model_args, data_args,
     training_args) = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(model_args), **vars(data_args),
                              **vars(training_args))
    # load model and tokenizer
    logging.warning('Loading model and tokenizer...')
    model, tokenizer = load_model_tokenizer(args=args)
    logging.warning('Successfully loaded model and tokenizer.')

    if 'llama' in args.model_name_or_path or 'baichuan' in args.model_name_or_path:
        logging.warning(
            f'Adding special tokens for {args.model_name_or_path}.')
        add_special_tokens_if_missing(tokenizer, model)

    # Create the training dataset and data collator
    logging.warning('Creating training dataset and data collator.')
    train_dataset = AlpacaDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Initialize the Trainer object and start training
    logging.warning('Initializing Trainer object.')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    logging.warning('Start Training...')
    if list(pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    logging.warning(f'Saving Model to {training_args.output_dir}')
    trainer.save_state()
    # Save the trained model
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)

    logging.warning('Done.')


if __name__ == '__main__':
    train()
