import os
from typing import Dict

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def get_server_model(args: Dict) -> torch.nn.Module:
    """
    Returns a language model for text generation that can be trained with mixed precision.

    Args:
        args (Dict): A dictionary containing various hyperparameters.
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

    print(f'Loading base model {args.model_name_or_path}...')
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
            print('=' * 80)
            print(
                'Your GPU supports bfloat16, you can accelerate training with the argument --bf16'
            )
            print('=' * 80)

    # Enable model parallelism.
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = torch_dtype

    # Load pre-trained adapters from checkpoint directory.
    print('Loading adapters from checkpoint.')
    model = PeftModel.from_pretrained(model,
                                      args.lora_model_name_or_path,
                                      is_trainable=False)

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
