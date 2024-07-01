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
from llamatuner.utils.model_utils import get_peft_state_maybe_zero_3

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
    model_args: ModelArguments,
    training_args: TrainingArguments,
    finetune_args: FinetuningArguments,
    logger: logging.Logger,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a pre-trained model and tokenizer for natural language processing
    tasks.

    Args:
        args: An object containing the input arguments.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """

    # Determine torch dtype for model based on arguments
    if training_args.fp16:
        torch_dtype = torch.float16
    elif training_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    device_map: Union[str, None] = 'auto'
    if finetune_args.use_qlora:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        device_map = ({
            '': int(os.environ.get('LOCAL_RANK') or 0)
        } if world_size != 1 else None)
        if len(finetune_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled(
        ):
            logger.info(
                'FSDP and ZeRO3 are both currently incompatible with QLoRA.')

    # Set configuration kwargs for tokenizer.
    config_kwargs = {
        'cache_dir': model_args.cache_dir,
        'trust_remote_code': model_args.trust_remote_code,
    }

    # Load the pre-trained model
    logger.info(f'Loading Model from {model_args.model_name_or_path}...')
    if finetune_args.quant_bit == 4:
        load_in_4bit = True
    elif finetune_args.quant_bit == 8:
        load_in_8bit = True

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

    # Add LoRA sparsity if specified
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
        logger.info('Preparemodel for kbit training!!!')
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=model_args.use_gradient_checkpointing)

        if torch.cuda.device_count() > 1:
            # Keeps Trainer from trying its own DataParallelism when more than 1 GPU is available
            setattr(model, 'model_parallel', True)
            setattr(model, 'is_parallelizable', True)

    logger.info('Get the get peft model...')
    model = get_peft_model(model, lora_config)

    if model_args.use_gradient_checkpointing:
        logger.info('Using gradient checkpointing...')
        model.enable_input_require_grads()
        model.config.use_cache = False  # Turn off when gradient checkpointing is enabled

    # Load the tokenizer
    logger.info(f'Loading tokenizer from {model_args.model_name_or_path}...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side='right',
        use_fast=False,
        model_max_length=model_args.model_max_length,
        **config_kwargs,
    )
    tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    finetune_args: FinetuningArguments,
    generating_args: GeneratingArguments,
) -> None:
    """Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.
        lora_args (LoraArguments): The arguments for the LoRA configuration.

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

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log
    output_dir = get_outdir(training_args.output_dir,
                            finetune_args.wandb_run_name)
    training_args.output_dir = get_outdir(output_dir, 'checkpoints')
    log_name = os.path.join(finetune_args.wandb_run_name,
                            timestamp).replace(os.path.sep, '_')
    log_file = os.path.join(output_dir, log_name + '.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    # Log on each process the small summary:
    logger.info(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
    )
    logger.info(
        f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    # load model and tokenizer
    model, tokenizer = load_model_tokenizer(model_args,
                                            training_args,
                                            finetune_args,
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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == 'right' else None,
        label_pad_token_id=IGNORE_INDEX
        if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Init the wandb
    logger.info('Initializing wandb...')
    wandb.init(
        dir=output_dir,
        project=finetune_args.wandb_project,
        name=finetune_args.wandb_run_name,
        tags=['lora-finetune', 'sft'],
        group='lora-finetune',
        config=args,
    )

    # Create a Trainer object and start training
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
    if training_args.resume_from_checkpoint and list(
            pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        logger.info('Resuming from checkpoint...')
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
                                                 finetune_args.lora_bias)

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == '__main__':
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        FinetuningArguments,
        GeneratingArguments,
    ))
    (model_args, data_args, training_args, finetune_args,
     generating_args) = (parser.parse_args_into_dataclasses())
    train(model_args, data_args, training_args, finetune_args, generating_args)
