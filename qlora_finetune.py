import argparse
import os
import time
from typing import Dict, Optional

import torch
import transformers
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig, Trainer,
                          set_seed)

from chatllms.data.data_utils import make_data_module
from chatllms.data.sft_dataset import (DataCollatorForSupervisedDataset,
                                       SupervisedDataset)
from chatllms.utils.callbacks import MMLUEvalCallback, SampleGenerateCallback
from chatllms.utils.config import (DataArguments, GenerationArguments,
                                   LoraArguments, ModelArguments,
                                   QuantArgments, TrainingArguments)
from chatllms.utils.logging import get_root_logger
from chatllms.utils.model_utils import (SavePeftModelCallback,
                                        add_special_tokens_if_missing,
                                        find_all_linear_names,
                                        get_last_checkpoint,
                                        print_trainable_parameters,
                                        verify_dtypes)
from chatllms.utils.training import predict_and_save, train_and_evaluate

torch.backends.cuda.matmul.allow_tf32 = True


def get_accelerate_model(args: Dict, checkpoint_dir: Optional[str],
                         logger: None) -> torch.nn.Module:
    """
    Returns a language model for text generation that can be trained with mixed precision.

    Args:
        args (Dict): A dictionary containing various hyperparameters.
        checkpoint_dir (str, optional): A directory containing pre-trained adapters for the model.

    Returns:
        torch.nn.Module: An instance of the language model.
    """
    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = 'auto'

    # If we are in a distributed setting, we need to set the device map and max memory per device.
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    # Check if we are doing full finetuning.
    if args.full_finetune: assert args.bits in [16, 32]

    logger.info(f'Loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else
                     (torch.bfloat16 if args.bf16 else torch.float32))
    torch_dtype = (torch.float32 if args.fp16 else
                   (torch.bfloat16 if args.bf16 else torch.float32))
    # Load the model.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type  # {'fp4', 'nf4'}
        ),
        torch_dtype=torch_dtype,
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Print a message if the GPU supports bfloat16.
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info('=' * 80)
            logger.info(
                'Your GPU supports bfloat16, you can accelerate training with the argument --bf16'
            )
            logger.info('=' * 80)

    # Enable model parallelism.
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = torch_dtype

    # Prepare the model for k-bit training if specified.
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing)

    # Enable gradient checkpointing if specified.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            # Load pre-trained adapters from checkpoint directory.
            logger.info('Loading adapters from checkpoint.')
            model = PeftModel.from_pretrained(model,
                                              os.path.join(
                                                  checkpoint_dir,
                                                  'adapter_model'),
                                              is_trainable=True)
        else:
            # Add LoRA modules to the model.
            logger.info('Adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias='none',
                task_type='CAUSAL_LM',
            )
            model = get_peft_model(model, config)

    # Convert certain model modules to a different precision as specified by the hyperparameters.
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments,
         QuantArgments, GenerationArguments))
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        quant_args,
        generation_args,
        extra_args,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))

    args = argparse.Namespace(**vars(model_args), **vars(data_args),
                              **vars(training_args), **vars(lora_args),
                              **vars(quant_args))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_file = os.path.join(args.output_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    logger.info('Training/evaluation parameters %s', args)
    # Check if training was already completed.
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        logger.info('Detected that training was already completed!')

    model = get_accelerate_model(args, checkpoint_dir, logger)
    model.config.use_cache = False
    print_trainable_parameters(args, model)
    logger.info('loaded model')
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side='right',
        use_fast=False,  # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else
        None,  # Needed for HF name change
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
    )

    # LLaMA tokenizer may not have correct special tokens set.
    # Check and add them if missing to prevent them from being parsed into different tokens.
    # Note that these are present in the vocabulary.
    # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
    logger.info('Adding special tokens.')
    if 'llama' in args.model_name_or_path or 'baichuan' in args.model_name_or_path:
        add_special_tokens_if_missing(tokenizer, model)

    dataset_dict = make_data_module(args)
    train_dataset = SupervisedDataset(
        dataset_dict['train'],
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    ) if args.do_train else None

    eval_dataset = SupervisedDataset(
        dataset_dict['eval'],
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    ) if args.do_eval else None

    predict_dataset = SupervisedDataset(
        dataset_dict['predict'],
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    ) if args.do_predict else None

    print(train_dataset, eval_dataset, predict_dataset)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, predict_with_generate=args.predict_with_generate)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    # Add callback to save adapter model.
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Add callback to generate samples.
    if args.sample_generate:
        trainer.add_callback(
            SampleGenerateCallback(
                tokenizer=tokenizer,
                generation_config=GenerationConfig(**vars(generation_args)),
                logger=logger,
            ))

    if args.do_mmlu_eval:
        eval_callback = MMLUEvalCallback(
            trainer=trainer,
            tokenizer=tokenizer,
            data_dir='./data',
            args=args,
        )
        trainer.add_callback(eval_callback)

    # Verify dtypes
    verify_dtypes(model)
    assert args.do_train or args.do_eval or args.do_predict
    if args.do_train or args.do_eval:
        train_and_evaluate(trainer, args, logger)
    if args.do_predict:
        predict_and_save(trainer, tokenizer, predict_dataset, args, logger)


if __name__ == '__main__':
    main()
