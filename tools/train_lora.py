import argparse
import logging
import math
import os
import pathlib
import sys
import time
from typing import Tuple, Union

import torch
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForSeq2Seq,
                          HfArgumentParser, PreTrainedModel,
                          PreTrainedTokenizer)
from transformers import Seq2SeqTrainingArguments as TrainingArguments
from transformers import Trainer, deepspeed

sys.path.append(os.getcwd())
from llamatuner.configs import (DataArguments, FinetuningArguments,
                                GeneratingArguments, ModelArguments)
from llamatuner.data.data_loader import get_dataset
from llamatuner.data.utils import split_dataset
from llamatuner.model.callbacks import ComputeMetrics
from llamatuner.model.utils.misc import find_all_linear_modules
from llamatuner.utils.constants import IGNORE_INDEX
from llamatuner.utils.logger_utils import get_outdir, get_root_logger
from llamatuner.utils.model_utils import (get_logits_processor,
                                          get_peft_state_maybe_zero_3)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def load_model_tokenizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    finetune_args: FinetuningArguments,
    logger: logging.Logger,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pre-trained model and tokenizer for natural language processing tasks.

    Args:
        model_args (ModelArguments): Arguments for the model configuration.
        training_args (TrainingArguments): Arguments for the training configuration.
        finetune_args (FinetuningArguments): Arguments for the finetuning configuration.
        logger (logging.Logger): Logger object for logging information.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer.
    """

    torch_dtype = (torch.float16 if training_args.fp16 else
                   torch.bfloat16 if training_args.bf16 else torch.float32)
    device_map: Union[str, None] = 'auto'

    if finetune_args.use_qlora:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        device_map = ({
            '': int(os.environ.get('LOCAL_RANK') or 0)
        } if world_size != 1 else None)
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled(
        ):
            logger.info(
                'FSDP and ZeRO3 are both currently incompatible with QLoRA.')

    config_kwargs = {
        'cache_dir': model_args.cache_dir,
        'trust_remote_code': model_args.trust_remote_code,
    }

    logger.info(f'Loading Model from {model_args.model_name_or_path}...')

    load_in_4bit = finetune_args.quant_bit == 4
    load_in_8bit = finetune_args.quant_bit == 8

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            llm_int8_threshold=finetune_args.llm_int8_threshold,
            llm_int8_has_fp16_weight=finetune_args.llm_int8_has_fp16_weight,
            bnb_4bit_use_double_quant=finetune_args.double_quant,
            bnb_4bit_quant_type=finetune_args.quant_type,
            bnb_4bit_compute_dtype=torch_dtype,
        ) if finetune_args.use_qlora else None,
        torch_dtype=torch_dtype,
        **config_kwargs,
    )

    logger.info('Adding LoRA modules...')
    if len(finetune_args.lora_target
           ) == 1 and finetune_args.lora_target[0] == 'all':
        target_modules = find_all_linear_modules(
            model, finetune_args.freeze_vision_tower)
    else:
        target_modules = finetune_args.lora_target

    lora_config = LoraConfig(
        r=finetune_args.lora_rank,
        lora_alpha=finetune_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=finetune_args.lora_dropout,
        bias=finetune_args.lora_bias,
        task_type='CAUSAL_LM',
    )

    if finetune_args.use_qlora:
        logger.info('Preparing model for kbit training...')
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing)

        if torch.cuda.device_count() > 1:
            setattr(model, 'model_parallel', True)
            setattr(model, 'is_parallelizable', True)

    logger.info('Getting the PEFT model...')
    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        logger.info('Using gradient checkpointing...')
        model.enable_input_require_grads()
        model.config.use_cache = False

    logger.info(f'Loading tokenizer from {model_args.model_name_or_path}...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side=model_args.padding_side,
        model_max_length=model_args.model_max_length,
        use_fast=False,
        **config_kwargs,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    finetune_args: FinetuningArguments,
    generating_args: GeneratingArguments,
) -> None:
    """
    Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.
        finetune_args (FinetuningArguments): The arguments for the finetuning configuration.
        generating_args (GeneratingArguments): The arguments for the generating configuration.

    Returns:
        None
    """

    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
        **vars(finetune_args),
        **vars(generating_args),
    )

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_dir = get_outdir(training_args.output_dir,
                            finetune_args.wandb_run_name)
    training_args.output_dir = get_outdir(output_dir, 'checkpoints')
    log_name = os.path.join(finetune_args.wandb_run_name,
                            timestamp).replace(os.path.sep, '_')
    log_file = os.path.join(output_dir, log_name + '.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    logger.info(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
    )
    logger.info(
        f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )

    logger.info('Loading model and tokenizer...')
    model, tokenizer = load_model_tokenizer(model_args,
                                            training_args,
                                            finetune_args,
                                            logger=logger)
    logger.info('Successfully loaded model and tokenizer.')

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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == 'right' else None,
        label_pad_token_id=IGNORE_INDEX
        if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    training_args.generation_max_length = (training_args.generation_max_length
                                           or data_args.cutoff_len)
    training_args.generation_num_beams = (data_args.eval_num_beams or
                                          training_args.generation_num_beams)
    training_args.remove_unused_columns = (False
                                           if model_args.visual_inputs else
                                           training_args.remove_unused_columns)

    gen_kwargs = generating_args.to_dict()
    gen_kwargs['eos_token_id'] = [tokenizer.eos_token_id
                                  ] + tokenizer.additional_special_tokens_ids
    gen_kwargs['pad_token_id'] = tokenizer.pad_token_id
    gen_kwargs['logits_processor'] = get_logits_processor()

    logger.info('Initializing wandb...')
    wandb.init(
        dir=output_dir,
        project=finetune_args.wandb_project,
        name=finetune_args.wandb_run_name,
        tags=['lora-finetune', 'sft'],
        group='lora-finetune',
        config=args,
    )

    logger.info('Creating a Trainer...')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(tokenizer)
        if training_args.predict_with_generate else None,
        **data_module,
    )

    logger.info('Starting training...')
    if training_args.do_train:
        if training_args.resume_from_checkpoint and list(
                pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
            logger.info(
                f'Resuming training from checkpoint {training_args.resume_from_checkpoint}'
            )
            train_result = trainer.train(resume_from_checkpoint=True)
        else:
            logger.info('Starting training from scratch...')
            train_result = trainer.train()

        if deepspeed.is_deepspeed_zero3_enabled():
            state_dict_zero3 = (
                trainer.model_wrapped._zero3_consolidated_16bit_state_dict())
            if training_args.local_rank == 0:
                state_dict = state_dict_zero3
        else:
            state_dict = get_peft_state_maybe_zero_3(model.named_parameters(),
                                                     finetune_args.lora_bias)

        if training_args.local_rank == 0:
            model.save_pretrained(training_args.output_dir,
                                  state_dict=state_dict)

        trainer.log_metrics('train', train_result.metrics)
        trainer.save_metrics('train', train_result.metrics)
        trainer.save_state()

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


if __name__ == '__main__':
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        FinetuningArguments,
        GeneratingArguments,
    ))
    model_args, data_args, training_args, finetune_args, generating_args = (
        parser.parse_args_into_dataclasses())
    train(model_args, data_args, training_args, finetune_args, generating_args)
