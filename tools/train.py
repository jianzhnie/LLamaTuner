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
                          DataCollatorForSeq2Seq, HfArgumentParser,
                          PreTrainedModel, PreTrainedTokenizer)
from transformers import Seq2SeqTrainingArguments as TrainingArguments
from transformers import Trainer

sys.path.append(os.getcwd())
from llamatuner.configs import (DataArguments, FinetuningArguments,
                                GeneratingArguments, ModelArguments)
from llamatuner.data.data_loader import get_dataset
from llamatuner.data.utils import split_dataset
from llamatuner.model.callbacks import ComputeMetrics
from llamatuner.utils.constants import IGNORE_INDEX
from llamatuner.utils.logger_utils import get_logger, get_outdir
from llamatuner.utils.model_utils import get_logits_processor

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def load_model_tokenizer(
    model_args: ModelArguments, text_logger: logging.Logger
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a pre-trained model and tokenizer for natural language processing
    tasks.

    Args:
        args: An object containing the input arguments.

    Returns:
        A tuple containing the loaded model and tokenizer.
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
    text_logger.info(f'Loading Model from {model_args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                 config=config,
                                                 **config_kwargs)

    # Enable model parallelism
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if not model_args.disable_gradient_checkpointing:
        text_logger.info('Using gradient checkpointing...')
        model.enable_input_require_grads()
        model.config.use_cache = (
            False  # Turn off when gradient checkpointing is enabled
        )

    # Load the tokenizer
    text_logger.info(
        f'Loading tokenizer from {model_args.model_name_or_path}...')
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


def train() -> None:
    """Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.

    Returns:
        None
    """
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        FinetuningArguments,
        GeneratingArguments,
    ))
    (model_args, data_args, training_args, finetune_args,
     generating_args) = (parser.parse_args_into_dataclasses())
    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
        **vars(finetune_args),
        **vars(generating_args),
    )

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # Set up the output directory
    output_dir = get_outdir(args.output_dir, args.wandb_run_name)
    training_args.output_dir = get_outdir(output_dir, 'checkpoints')
    log_name = os.path.join(args.wandb_run_name,
                            timestamp).replace(os.path.sep, '_')
    log_file = os.path.join(output_dir, log_name + '.log')
    text_logger = get_logger(name='llamatuner',
                             log_file=log_file,
                             log_level='INFO')

    # load model and tokenizer
    text_logger.info('Loading model and tokenizer...')
    model, tokenizer = load_model_tokenizer(model_args,
                                            text_logger=text_logger)
    text_logger.info('Successfully loaded model and tokenizer.')

    # Create a supervised dataset and Trainer, then train the model
    text_logger.info('Creating a supervised dataset and DataCollator...')

    all_dataset = get_dataset(
        data_args,
        model_args,
        training_args,
        stage='sft',
        tokenizer=tokenizer,
        processor=None,
    )
    data_module = split_dataset(all_dataset, data_args, training_args)
    text_logger.info('Successfully created the supervised dataset.')
    text_logger.info('Creating DataCollator for Seq2Seq...')
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == 'right' else None,
        label_pad_token_id=IGNORE_INDEX
        if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (training_args.generation_max_length
                                           or data_args.cutoff_len)
    training_args.generation_num_beams = (data_args.eval_num_beams or
                                          training_args.generation_num_beams)
    training_args.remove_unused_columns = (False
                                           if model_args.visual_inputs else
                                           training_args.remove_unused_columns)

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs['eos_token_id'] = [tokenizer.eos_token_id
                                  ] + tokenizer.additional_special_tokens_ids
    gen_kwargs['pad_token_id'] = tokenizer.pad_token_id
    gen_kwargs['logits_processor'] = get_logits_processor()

    # Init the wandb
    text_logger.info('Initializing wandb project...')
    wandb.init(
        dir=output_dir,
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=['full-finetune', 'sft'],
        group='full-finetune',
        config=args,
    )
    # Initialize the Trainer object and start training
    text_logger.info('Initializing Trainer object.')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(tokenizer)
        if training_args.predict_with_generate else None,
        **data_module,
    )
    # Training
    training_args: TrainingArguments = trainer.args
    if training_args.do_train:
        if (list(pathlib.Path(training_args.output_dir).glob('checkpoint-*'))
                and training_args.resume_from_checkpoint):
            text_logger.info('Resuming training from checkpoint %s' %
                             (training_args.resume_from_checkpoint))
            train_result = trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint)
        else:
            text_logger.info('Starting training from scratch...')
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

    text_logger.info('Done.')


if __name__ == '__main__':
    train()
