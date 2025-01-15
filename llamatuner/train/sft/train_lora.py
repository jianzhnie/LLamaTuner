import argparse
import logging
import math
import os
import pathlib
import sys
import time
from typing import Tuple, Union

import deepspeed
import torch
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForSeq2Seq,
                          HfArgumentParser, PreTrainedModel,
                          PreTrainedTokenizer)
from transformers import Seq2SeqTrainingArguments as TrainingArguments
from transformers import Trainer

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
                                          get_peft_state_maybe_zero_3,
                                          print_model_dtypes,
                                          print_trainable_parameters)

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

    # Determine torch dtype for model based on arguments
    torch_dtype = (torch.float32 if training_args.fp16 else
                   (torch.bfloat16 if training_args.bf16 else torch.float32))

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
        low_cpu_mem_usage=True,
        # BitsAndBytesConfig设置存储格式和计算格式，以及优化方式
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            llm_int8_threshold=finetune_args.llm_int8_threshold,  # int8的门限
            llm_int8_has_fp16_weight=finetune_args.
            llm_int8_has_fp16_weight,  # int8的LLM，是否包含fp16的权重
            bnb_4bit_use_double_quant=finetune_args.double_quant,  # 是否进行双重量化
            bnb_4bit_quant_type=finetune_args.quant_type,  # {'fp4', 'nf4'}
            bnb_4bit_compute_dtype=torch_dtype,  # 计算时使用的数据类型
        ) if finetune_args.use_qlora else None,
        torch_dtype=torch_dtype,
        **config_kwargs,
    )

    # Enable model parallelism.
    # 设置两个和并行操作相关的参数
    if torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 GPU is available
        setattr(model, 'model_parallel', True)
        setattr(model, 'is_parallelizable', True)

    # Prepare the model for k-bit training if specified.
    if finetune_args.use_qlora:
        logger.info('Preparemodel for kbit training!!!')
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing)
    # Print a message if the GPU supports bfloat16.
    # 如果计算类型为 torch.float16 并且 args.bits==4，也就是4bit量化模型时，进行如下操作。
    if torch_dtype == torch.float16 and finetune_args.quant_bit.bits == 4:
        # 得到显卡的计算能力的最大值和最小值，分别对应major和minor
        # 只有major >= 8时的GPU才支持bfloat16格式，可以使用参数--bf16来加速训练
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info(
                'Your GPU supports bfloat16, you can accelerate training with the argument --bf16'
            )

    # Add LoRA sparsity if specified
    logger.info('Adding LoRA modules...')
    if len(finetune_args.lora_target
           ) == 1 and finetune_args.lora_target[0] == 'all':
        target_modules = find_all_linear_modules(
            model, finetune_args.freeze_vision_tower)
    else:
        target_modules = finetune_args.lora_target

    lora_config = LoraConfig(
        r=finetune_args.lora_rank,  # lora层A矩阵的列大小和B矩阵的行大小
        lora_alpha=finetune_args.lora_alpha,  # 缩放因子
        target_modules=target_modules,  # 需要进行lora网络操作的模块名称列表
        lora_dropout=finetune_args.lora_dropout,  # 是否使用dropout, 正则化操作
        bias=finetune_args.lora_bias,  # 是否对偏差参数进行处理
        task_type='CAUSAL_LM',  # 模型名称，一种标记
    )

    logger.info('Getting the PEFT model...')
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing if specified
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


def run_lora_sft(
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

    logger.info('Printing trainable parameters...')
    print_trainable_parameters(model, kbit=finetune_args.quant_bit)

    # Verify dtypes
    logger.info('Print model dtypes...')
    print_model_dtypes(model)

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

        metrics = train_result.metrics
        metrics['train_samples'] = len(trainer.train_dataset)
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix='eval')
        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float('inf')

        metrics['perplexity'] = perplexity
        metrics['eval_samples'] = len(trainer.eval_dataset)
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
    run_lora_sft(model_args, data_args, training_args, finetune_args,
                 generating_args)
