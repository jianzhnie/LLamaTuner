import argparse
import logging
import math
import os
import pathlib
import sys
import time
from typing import Tuple

import wandb
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, PreTrainedModel,
                          PreTrainedTokenizer)
from transformers import Seq2SeqTrainingArguments as TrainingArguments
from transformers import Trainer

sys.path.append(os.getcwd())
from llamatuner.configs import (DataArguments, FinetuningArguments,
                                ModelArguments)
from llamatuner.data.data_loader import get_dataset
from llamatuner.data.utils import split_dataset
from llamatuner.utils.logger_utils import get_logger, get_outdir


def load_model_tokenizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    logger: logging.Logger,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a pre-trained model and tokenizer for natural language processing tasks.

    Args:
        model_args (ModelArguments): Arguments for the model configuration.
        training_args (TrainingArguments): Arguments for the training configuration.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.
    """
    config_kwargs = {
        'cache_dir': model_args.cache_dir,
        'trust_remote_code': model_args.trust_remote_code,
    }

    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                        **config_kwargs)

    orig_ctx_len = getattr(config, 'max_position_embeddings', None)
    if orig_ctx_len and model_args.model_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(model_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {'type': 'linear', 'factor': scaling_factor}
    config.use_cache = False

    # Load the pre-trained model
    logger.info(f'Loading Model from {model_args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                 config=config,
                                                 **config_kwargs)

    # Enable model parallelism
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if training_args.gradient_checkpointing:
        logger.info('Using gradient checkpointing...')
        model.enable_input_require_grads()
        model.config.use_cache = (
            False  # Turn off when gradient checkpointing is enabled
        )

    # Load the tokenizer
    logger.info(f'Loading tokenizer from {model_args.model_name_or_path}...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side=model_args.padding_side,
        model_max_length=model_args.model_max_length,
        use_fast=False,
        **config_kwargs,
    )
    # Add special tokens if they are missing
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer


def run_pt(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    finetune_args: FinetuningArguments,
) -> None:

    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
        **vars(finetune_args),
    )
    # Initialize the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # Set up the output directory
    output_dir = get_outdir(training_args.output_dir,
                            finetune_args.wandb_run_name)
    training_args.output_dir = get_outdir(output_dir, 'checkpoints')
    log_name = os.path.join(finetune_args.wandb_run_name,
                            timestamp).replace(os.path.sep, '_')
    log_file = os.path.join(output_dir, log_name + '.log')
    logger = get_logger(name='llamatuner', log_file=log_file, log_level='INFO')

    # Load model and tokenizer
    logger.info('Loading model and tokenizer...')
    model, tokenizer = load_model_tokenizer(model_args,
                                            training_args,
                                            logger=logger)
    logger.info('Successfully loaded model and tokenizer.')

    # Create a supervised dataset and Trainer, then train the model
    logger.info('Creating a supervised dataset and DataCollator...')

    all_dataset = get_dataset(
        data_args,
        model_args,
        training_args,
        stage='sft',
        tokenizer=tokenizer,
        processor=None,
    )
    data_module = split_dataset(all_dataset, data_args, training_args)
    logger.info('Successfully created the supervised dataset.')
    logger.info('Creating DataCollator for Seq2Seq...')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)
    # Initialize wandb
    logger.info('Initializing wandb project...')
    wandb.init(
        dir=output_dir,
        project=finetune_args.wandb_project,
        name=finetune_args.wandb_run_name,
        tags=['full-finetune', 'sft'],
        group='full-finetune',
        config=args,
    )
    # Initialize the Trainer object and start training
    logger.info('Initializing Trainer object.')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        **data_module,
    )
    # Training
    if training_args.do_train:
        if (list(pathlib.Path(training_args.output_dir).glob('checkpoint-*'))
                and training_args.resume_from_checkpoint):
            logger.info('Resuming training from checkpoint %s' %
                        (training_args.resume_from_checkpoint))
            train_result = trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint)
        else:
            logger.info('Starting training from scratch...')
            train_result = trainer.train()

        trainer.log_metrics('train', train_result.metrics)
        trainer.save_metrics('train', train_result.metrics)
        trainer.save_state()
        trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix='eval')
        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float('inf')

        metrics['perplexity'] = perplexity
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    logger.info('Done.')
