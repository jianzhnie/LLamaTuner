import argparse
import logging
import math
import os
import pathlib
import sys
import time
from typing import Tuple

import torch
import wandb
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, PreTrainedModel,
                          PreTrainedTokenizer, Trainer)

sys.path.append(os.getcwd())
from chatllms.configs import DataArguments, ModelArguments, TrainingArguments
from chatllms.data import make_supervised_data_module
from chatllms.utils.logger_utils import get_outdir, get_root_logger

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def load_model_tokenizer(
    args, text_logger: logging.Logger
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a pre-trained model and tokenizer for natural language processing
    tasks.

    Args:
        args: An object containing the input arguments.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    # Determine the torch data type based on the input arguments
    torch_dtype = (torch.float16 if args.fp16 else
                   (torch.bfloat16 if args.bf16 else torch.float32))

    config_kwargs = {
        'cache_dir': args.cache_dir,
        'trust_remote_code': args.trust_remote_code,
    }

    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        **config_kwargs)

    orig_ctx_len = getattr(config, 'max_position_embeddings', None)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
        config.rope_scaling = {'type': 'linear', 'factor': scaling_factor}
    config.use_cache = False

    # Load the pre-trained model
    text_logger.info(f'Loading Model from {args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        **config_kwargs,
    )

    # Enable model parallelism
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if args.gradient_checkpointing:
        text_logger.info('Using gradient checkpointing...')
        model.enable_input_require_grads()
        model.config.use_cache = (
            False  # Turn off when gradient checkpointing is enabled
        )

    # Load the tokenizer
    text_logger.info(f'Loading tokenizer from {args.model_name_or_path}...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side=args.padding_side,
        model_max_length=args.model_max_length,
        use_fast=False,
        **config_kwargs,
    )
    # Add special tokens if they are missing
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer


def train() -> None:
    """Trains a language model using Hugging Face's Transformers library.

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
    data_args.init_for_training()
    args = argparse.Namespace(**vars(model_args), **vars(data_args),
                              **vars(training_args))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log
    output_dir = get_outdir(args.output_dir, args.wandb_run_name)
    training_args.output_dir = get_outdir(output_dir, 'checkpoints')
    wandb_dir = get_outdir(output_dir, 'wandb')
    log_name = os.path.join(args.wandb_run_name,
                            timestamp).replace(os.path.sep, '_')
    log_file = os.path.join(output_dir, log_name + '.log')
    text_logger = get_root_logger(log_file=log_file, log_level='INFO')

    # load model and tokenizer
    text_logger.info('Loading model and tokenizer...')
    model, tokenizer = load_model_tokenizer(args=args, text_logger=text_logger)
    text_logger.info('Successfully loaded model and tokenizer.')

    # Create a supervised dataset and Trainer, then train the model
    text_logger.info('Creating a supervised dataset and DataCollator...')
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              text_logger=text_logger,
                                              args=args)

    # Initialize the Trainer object and start training
    text_logger.info('Initializing Trainer object.')

    # Init the wandb
    wandb.init(
        dir=wandb_dir,
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=['full-finetune', 'sft'],
        group='full-finetune',
        config=args,
    )
    # Start trainner
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    # Training
    if args.do_train:
        if (list(pathlib.Path(args.output_dir).glob('checkpoint-*'))
                and args.resume_from_checkpoint):
            train_result = trainer.train(
                resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            train_result = trainer.train()

        trainer.log_metrics('train', train_result.metrics)
        trainer.save_metrics('train', train_result.metrics)
        trainer.save_state()
        trainer.save_model()

    # Evaluation
    if args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix='eval')
        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float('inf')

        metrics['perplexity'] = perplexity
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    text_logger.info('Done.')


if __name__ == '__main__':
    train()
