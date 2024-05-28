import argparse
import logging
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import torch
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          PreTrainedModel, PreTrainedTokenizer, Trainer,
                          deepspeed)

sys.path.append(os.getcwd())
from chatllms.configs import DataArguments, ModelArguments, TrainingArguments
from chatllms.data import make_supervised_data_module
from chatllms.utils.logger_utils import get_outdir, get_root_logger
from chatllms.utils.model_utils import get_peft_state_maybe_zero_3

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj'])
    lora_weight_path: str = ''
    lora_bias: str = 'none'
    q_lora: bool = False


# Borrowed from peft.utils.get_peft_model_state_dict
def load_model_tokenizer(
    args: argparse.Namespace, text_logger: logging.Logger
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a pre-trained model and tokenizer for natural language processing
    tasks.

    Args:
        args: An object containing the input arguments.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """

    # Determine torch dtype for model based on arguments
    if args.fp16:
        compute_dtype = torch.float16
    elif args.bf16:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    device_map: Union[str, None] = 'auto'
    if args.q_lora:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        device_map = ({
            '': int(os.environ.get('LOCAL_RANK') or 0)
        } if world_size != 1 else None)
        if len(args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            text_logger.info(
                'FSDP and ZeRO3 are both currently incompatible with QLoRA.')

    # Set configuration kwargs for tokenizer.
    config_kwargs = {
        'cache_dir': args.cache_dir,
        'trust_remote_code': args.trust_remote_code,
    }

    # Load the pre-trained model
    text_logger.info(f'Loading Model from {args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
        ) if args.q_lora else None,
        torch_dtype=compute_dtype,
        **config_kwargs,
    )

    # Add LoRA sparsity if specified
    text_logger.info('Adding LoRA modules...')
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type='CAUSAL_LM',
    )
    if args.q_lora:
        text_logger.info('Preparemodel for kbit training!!!')
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing)

        if torch.cuda.device_count() > 1:
            # Keeps Trainer from trying its own DataParallelism when more than 1 GPU is available
            setattr(model, 'model_parallel', True)
            setattr(model, 'is_parallelizable', True)

    text_logger.info('Get the get peft model...')
    model = get_peft_model(model, lora_config)

    if args.deepspeed is not None and args.local_rank == 0:
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        text_logger.info('Using gradient checkpointing...')
        model.enable_input_require_grads()
        model.config.use_cache = False  # Turn off when gradient checkpointing is enabled

    # Load the tokenizer
    text_logger.info(f'Loading tokenizer from {args.model_name_or_path}...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='right',
        use_fast=False,
        model_max_length=args.model_max_length,
        **config_kwargs,
    )
    tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer


def train() -> None:
    """Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.
        lora_args (LoraArguments): The arguments for the LoRA configuration.

    Returns:
        None
    """
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses(
    )
    data_args.init_for_training()
    args = argparse.Namespace(**vars(model_args), **vars(data_args),
                              **vars(training_args), **vars(lora_args))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log
    output_dir = get_outdir(args.output_dir, args.wandb_run_name)
    training_args.output_dir = get_outdir(output_dir, 'checkpoints')
    log_name = os.path.join(args.wandb_run_name,
                            timestamp).replace(os.path.sep, '_')
    log_file = os.path.join(output_dir, log_name + '.log')
    text_logger = get_root_logger(log_file=log_file, log_level='INFO')

    # Log on each process the small summary:
    text_logger.info(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
    )
    text_logger.info(
        f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    # load model and tokenizer
    model, tokenizer = load_model_tokenizer(args=args, text_logger=text_logger)
    text_logger.info('Successfully loaded model and tokenizer.')

    # Create a supervised dataset and Trainer, then train the model
    text_logger.info('Creating a supervised dataset and DataCollator...')
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              args=args,
                                              text_logger=text_logger)

    # Init the wandb
    text_logger.info('Initializing wandb...')
    wandb.init(
        dir=output_dir,
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=['lora-finetune', 'sft'],
        group='lora-finetune',
        config=args,
    )

    # Create a Trainer object and start training
    text_logger.info('Creating a Trainer...')
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      **data_module)

    text_logger.info('Starting training...')
    if training_args.resume_from_checkpoint and list(
            pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        text_logger.info('Resuming from checkpoint...')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    # Save the trained model
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict(
        )
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(),
                                                 lora_args.lora_bias)

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == '__main__':
    train()
