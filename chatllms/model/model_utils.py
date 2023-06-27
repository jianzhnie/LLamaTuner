import os
from typing import Dict, Optional, Tuple

import torch
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from chatllms.utils.model_utils import find_all_linear_names

check_min_version('4.29.1')


def load_model_tokenizer(
        args: Dict, checkpoint_dir: Optional[str],
        logger: None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
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

    config_kwargs = {
        'cache_dir': args.cache_dir,
        'revision': args.model_revision,
        'use_auth_token': args.use_auth_token,
        'trust_remote_code': args.trust_remote_code,
    }

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='right',
        use_fast=False,
        # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,
        # Needed for HF name change
        **config_kwargs)

    # Check if we are doing full finetuning.
    if args.full_finetune: assert args.bits in [16, 32]

    logger.info(f'Loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else
                     (torch.bfloat16 if args.bf16 else torch.float32))
    torch_dtype = (torch.fp16 if args.fp16 else
                   (torch.bfloat16 if args.bf16 else torch.float32))

    # Quantization configurations (using bitsandbytes library).
    if args.bits == 8:
        require_version('bitsandbytes>=0.37.0',
                        'To fix: pip install bitsandbytes>=0.37.0')
        config_kwargs['load_in_8bit'] = True
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    elif args.bits == 4:
        require_version('bitsandbytes>=0.39.0',
                        'To fix: pip install bitsandbytes>=0.39.0')
        require_version('transformers>=4.30.1',
                        'To fix: pip install transformers>=4.30.1')
        require_version('accelerate>=0.20.3',
                        'To fix: pip install accelerate>=0.20.3')
        require_version(
            'peft>=0.4.0.dev0',
            'To fix: pip install git+https://github.com/huggingface/peft.git')

        config_kwargs['load_in_4bit'] = True
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=args.bits is True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        )

    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 device_map=device_map,
                                                 low_cpu_mem_usage=True,
                                                 **config_kwargs)

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
    return model, tokenizer
