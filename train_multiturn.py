import argparse
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers.trainer import Trainer

from chatllms.data.conv_dataset import make_supervised_data_module
from chatllms.utils.model_utils import (add_special_tokens_if_missing,
                                        safe_save_model_for_hf_trainer)
from train import load_model_tokenizer


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
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    model_max_length: int = field(
        default=2048,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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

    # Add special tokens if they are missing
    logging.warning('Creating training dataset and data collator.')
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        lazy_preprocess=args.lazy_preprocess,
        data_path=args.data_path)

    # Initialize the Trainer object and start training
    logging.warning('Initializing Trainer object.')
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      **data_module)
    if list(pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == '__main__':
    train()
