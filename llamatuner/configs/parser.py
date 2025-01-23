import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from transformers import HfArgumentParser
from transformers import Seq2SeqTrainingArguments as TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode

from llamatuner.configs.data_args import DataArguments
from llamatuner.configs.eval_args import EvaluationArguments
from llamatuner.configs.finetuning_args import FinetuningArguments
from llamatuner.configs.generating_args import GeneratingArguments
from llamatuner.configs.model_args import ModelArguments
from llamatuner.utils.constants import CHECKPOINT_NAMES
from llamatuner.utils.logger_utils import get_logger
from llamatuner.utils.misc import (check_dependencies, check_version,
                                   get_current_device)

logger = get_logger('llamatuner')

check_dependencies()

TRAIN_ARGS = [
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
]
TRAIN_CLS = Tuple[ModelArguments, DataArguments, TrainingArguments,
                  FinetuningArguments, GeneratingArguments, ]
INFER_ARGS = [
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
]
INFER_CLS = Tuple[ModelArguments, DataArguments, FinetuningArguments,
                  GeneratingArguments]
EVAL_ARGS = [
    ModelArguments,
    DataArguments,
    EvaluationArguments,
    FinetuningArguments,
]
EVAL_CLS = Tuple[ModelArguments, DataArguments, EvaluationArguments,
                 FinetuningArguments, ]


def parse_args(parser: HfArgumentParser,
               args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith('.yaml'):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    if unknown_args:
        logger.info(parser.format_help())
        logger.info(
            'Got unknown args, potentially deprecated arguments: {}'.format(
                unknown_args))
        raise ValueError(
            'Some specified arguments are not used by the HfArgumentParser: {}'
            .format(unknown_args))

    return (*parsed_args, )


def set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def verify_model_args(
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
) -> None:

    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != 'lora':
        raise ValueError('Adapter is only valid for the LoRA method.')

    if finetuning_args.use_qlora and finetuning_args.quant_bit is not None:
        if finetuning_args.finetuning_type != 'lora':
            raise ValueError('Quant is only compatible with the LoRA method.')

        if model_args.resize_vocab:
            raise ValueError(
                'Cannot resize embedding layers of a quantized model.')

        if (model_args.adapter_name_or_path is not None
                and len(model_args.adapter_name_or_path) != 1):
            raise ValueError(
                'Quantized model only accepts a single adapter. Merge them first.'
            )


def check_extra_dependencies(
    training_args: Optional[TrainingArguments] = None, ) -> None:

    if training_args is not None and training_args.predict_with_generate:
        check_version('jieba', mandatory=True)
        check_version('nltk', mandatory=True)
        check_version('rouge_chinese', mandatory=True)


def parse_train_args(args: Optional[Dict[str, Any]] = None) -> TRAIN_CLS:
    parser = HfArgumentParser(TRAIN_ARGS)
    return parse_args(parser, args)


def parse_infer_args(args: Optional[Dict[str, Any]] = None) -> INFER_CLS:
    parser = HfArgumentParser(INFER_ARGS)
    return parse_args(parser, args)


def parse_eval_args(args: Optional[Dict[str, Any]] = None) -> EVAL_CLS:
    parser = HfArgumentParser(EVAL_ARGS)
    return parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None) -> TRAIN_CLS:
    (
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
    ) = parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        set_transformers_logging()

    # Check arguments
    if finetuning_args.stage != 'pt' and data_args.template is None:
        raise ValueError('Please specify which `template` to use.')

    if finetuning_args.stage != 'sft':
        if training_args.predict_with_generate:
            raise ValueError(
                '`predict_with_generate` cannot be set as True except SFT.')

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError(
                '`train_on_prompt` or `mask_history` cannot be set as True except SFT.'
            )

    if finetuning_args.stage != 'sft' and training_args.do_predict and training_args.predict_with_generate:
        raise ValueError(
            '`predict_with_generate` cannot be set as True except SFT.')

    if finetuning_args.stage in ['rm', 'ppo'
                                 ] and training_args.load_best_model_at_end:
        raise ValueError(
            'RM and PPO stages do not support `load_best_model_at_end`.')

    if finetuning_args.stage == 'ppo' and not training_args.do_train:
        raise ValueError(
            'PPO training does not support evaluation, use the SFT stage to evaluate models.'
        )

    if (finetuning_args.stage == 'ppo' and training_args.report_to
            and training_args.report_to[0] not in ['wandb', 'tensorboard']):
        raise ValueError('PPO only accepts wandb or tensorboard logger.')

    if training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError(
            'Please launch distributed training with `llamatuner-cli` or `torchrun`.'
        )

    if (training_args.deepspeed
            and training_args.parallel_mode != ParallelMode.DISTRIBUTED):
        raise ValueError(
            'Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.')

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError('Please specify `max_steps` in streaming mode.')

    if training_args.do_train and data_args.dataset is None:
        raise ValueError('Please specify dataset for training.')

    if training_args.do_train and finetuning_args.quant_device_map == 'auto':
        raise ValueError(
            'Cannot use device map for quantized models in training.')

    verify_model_args(model_args, finetuning_args)
    check_extra_dependencies(model_args, finetuning_args, training_args)

    if (training_args.do_train and finetuning_args.finetuning_type == 'lora'
            and finetuning_args.quant_bit is None and model_args.resize_vocab
            and finetuning_args.additional_target is None):
        logger.warning(
            'Remember to add embedding layers to `additional_target` to make the added tokens trainable.'
        )

    if training_args.do_train and (not training_args.fp16) and (
            not training_args.bf16):
        logger.warning('We recommend enable mixed precision training.')

    if (not training_args.do_train) and finetuning_args.quant_bit is not None:
        logger.warning(
            'Evaluating model in 4/8-bit mode may cause lower scores.')

    if ((not training_args.do_train) and finetuning_args.stage == 'dpo'
            and finetuning_args.ref_model is None):
        logger.warning(
            'Specify `ref_model` for computing rewards at evaluation.')

    # Post-process training arguments
    if (training_args.parallel_mode == ParallelMode.DISTRIBUTED
            and training_args.ddp_find_unused_parameters is None
            and finetuning_args.finetuning_type == 'lora'):
        logger.warning(
            '`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.'
        )
        training_args.ddp_find_unused_parameters = False

    if finetuning_args.stage in ['rm', 'ppo'
                                 ] and finetuning_args.finetuning_type in [
                                     'full',
                                     'freeze',
                                 ]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning('Cannot resume from checkpoint in current stage.')
            training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if (training_args.resume_from_checkpoint is None and training_args.do_train
            and os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
            and can_resume_from_checkpoint):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(
                os.path.isfile(os.path.join(training_args.output_dir, name))
                for name in CHECKPOINT_NAMES):
            raise ValueError(
                'Output directory already exists and is not empty. Please set `overwrite_output_dir`.'
            )

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info('Resuming training from {}.'.format(
                training_args.resume_from_checkpoint))
            logger.info(
                'Change `output_dir` or use `overwrite_output_dir` to avoid.')

    if (finetuning_args.stage in ['rm', 'ppo']
            and finetuning_args.finetuning_type == 'lora'
            and training_args.resume_from_checkpoint is not None):
        logger.warning(
            'Add {} to `adapter_name_or_path` to resume training from checkpoint.'
            .format(training_args.resume_from_checkpoint))

    # Post-process model arguments
    if training_args.bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.device_map = {'': get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    data_args.packing = (data_args.packing if data_args.packing is not None
                         else finetuning_args.stage == 'pt')

    # Log on each process the small summary
    logger.info(
        'Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}'
        .format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            training_args.parallel_mode == ParallelMode.DISTRIBUTED,
            str(model_args.compute_dtype),
        ))

    transformers.set_seed(training_args.seed)

    return (
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
    )


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = parse_infer_args(
        args)
    set_transformers_logging()

    if data_args.template is None:
        raise ValueError('Please specify which `template` to use.')

    verify_model_args(model_args, finetuning_args)
    check_extra_dependencies(model_args, finetuning_args)

    model_args.device_map = 'auto'
    return model_args, data_args, finetuning_args, generating_args


def get_eval_args(args: Optional[Dict[str, Any]] = None) -> EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = parse_eval_args(args)

    set_transformers_logging()

    if data_args.template is None:
        raise ValueError('Please specify which `template` to use.')

    verify_model_args(model_args, finetuning_args)
    check_extra_dependencies(model_args, finetuning_args)

    model_args.device_map = 'auto'

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args
