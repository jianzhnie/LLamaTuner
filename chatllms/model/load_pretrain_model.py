import argparse
import os
from os.path import exists, join
from typing import Optional, Tuple

import torch
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from peft.utils import CONFIG_NAME, WEIGHTS_NAME
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from chatllms.utils.model_utils import (add_special_tokens_if_missing,
                                        find_all_linear_names)

check_min_version('4.29.1')


def load_model_tokenizer(
    args: argparse.Namespace = None,
    checkpoint_dir: Optional[str] = None,
    is_trainable: Optional[bool] = True,
    logger=None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Returns a language model and tokenizer for text generation that can be trained with mixed precision.
    Support both training and inference.

    Args:
        args: A dictionary containing various hyperparameters.
        checkpoint_dir: A directory containing pre-trained adapters for the model.
        is_trainable: A bool indicating whether the model can be trained or not.
        logger: A logger object to log messages during execution.
    Returns:
        A tuple containing an instance of the language model and an instance of the tokenizer.
    """

    # Log a warning message if the checkpoint is not found at evaluation time.
    if not is_trainable and checkpoint_dir is None:
        logger.warning(
            'Checkpoint is not found at evaluation, load the original model.')

    # Determine number of GPUs and max memory per device.
    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = 'auto'

    # If we are in a distributed setting, we need to set the device map and max memory per device.
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    # Set configuration kwargs for tokenizer.
    config_kwargs = {
        'cache_dir': args.cache_dir,
        'revision': args.model_revision,
        'use_auth_token': args.use_auth_token,
        'trust_remote_code': args.trust_remote_code,
    }

    # If full finetuning is enabled, check that bits are either 16 or 32.
    if args.full_finetune: assert args.bits in [16, 32]
    # Set quantization configurations using bitsandbytes library.
    if args.bits == 8:
        require_version('bitsandbytes>=0.37.0',
                        'To fix: pip install bitsandbytes>=0.37.0')
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

    # Set compute and torch dtypes based on hyperparameters.
    compute_dtype = (torch.float16 if args.fp16 else
                     (torch.bfloat16 if args.bf16 else torch.float32))

    # 实际的torch数据类型, args.fp16对应torch.float32, args.bf16对应torch.bfloat16, 否则为torch.float32
    torch_dtype = (torch.float32 if args.fp16 else
                   (torch.bfloat16 if args.bf16 else torch.float32))

    logger.info(f'Loading base model {args.model_name_or_path}...')
    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,  # 模型名称或路径
        device_map=device_map,  # 设备分配方案，可以使用auto，cpu，gpu，还可以使用字典指定每个设备的编号
        max_memory=max_memory,  # 显存分配方案，还可以使用cpu和硬盘上的内存映射
        low_cpu_mem_usage=True,  # 是否使用低内存占用的模式
        load_in_4bit=args.bits == 4,  # 是否使用4bit加载
        load_in_8bit=args.bits == 8,  # 是否使用8bit加载
        # BitsAndBytesConfig设置存储格式和计算格式，以及优化方式
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,  # int8的门限，这个是做什么的
            llm_int8_has_fp16_weight=False,  # int8的LLM，是否包含fp16的权重
            bnb_4bit_compute_dtype=compute_dtype,  # 计算时使用的数据类型
            bnb_4bit_use_double_quant=args.double_quant,  # 是否进行双重量化
            bnb_4bit_quant_type=args.quant_type  # {'fp4', 'nf4'}
        ),
        torch_dtype=torch_dtype,
        **config_kwargs,
    )
    # Print a message if the GPU supports bfloat16.
    # 如果计算类型为torch.float16并且args.bits==4，也就是4bit量化模型时，进行如下操作。
    if compute_dtype == torch.float16 and args.bits == 4:
        # 得到显卡的计算能力的最大值和最小值，分别对应major和minor
        # 只有major>=8时的GPU才支持bfloat16格式，可以使用参数--bf16来加速训练
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info('=' * 80)
            logger.info(
                'Your GPU supports bfloat16, you can accelerate training with the argument --bf16'
            )
            logger.info('=' * 80)

    # Enable model parallelism.
    # 设置两个和并行操作相关的参数
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = torch_dtype

    # Prepare the model for k-bit training if specified.
    if not args.full_finetune:
        # 如果不是全参数微调，那么使用如下函数将模型的参数再进行处理，方便进行int8数据格式的训练微调
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing)

    # Enable gradient checkpointing if specified.
    if args.gradient_checkpointing:
        # 是否使用梯度检查
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # turn off when gradient checkpointing is enabled

    if not args.full_finetune:
        if checkpoint_dir is not None:
            # Load pre-trained adapters from checkpoint directory.
            logger.info('Loading adapters from checkpoint... ')
            adapter_model_path = join(checkpoint_dir, 'adapter_model')
            assert exists(join(adapter_model_path, CONFIG_NAME)) and exists(
                join(adapter_model_path, WEIGHTS_NAME)), ValueError(
                    'The given checkpoint may be not a LoRA checkpoint')

            model = PeftModel.from_pretrained(model,
                                              adapter_model_path,
                                              is_trainable=is_trainable)
            logger.info(
                'Loaded fine-tuned model from checkpoint(s): {}'.format(
                    adapter_model_path))

        else:
            # Add LoRA modules to the model.
            logger.info('No checkpoint_dir founded, will init adapters...')
            logger.info('Adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,  # lora层A矩阵的列大小和B矩阵的行大小
                lora_alpha=args.lora_alpha,  # 缩放因子
                target_modules=modules,  # 需要进行lora网络操作的模块名称列表
                lora_dropout=args.lora_dropout,  # 是否使用dropout，一种正则化操作
                bias='none',  # 不对偏差参数进行处理
                task_type='CAUSAL_LM',  # 模型名称，一种标记
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

    if not is_trainable:
        model.requires_grad_(False)

    # Instantiate tokenizer.
    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='right',
        use_fast=False,
        model_max_length=args.model_max_length,
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,
        **config_kwargs,
    )
    # LLaMA tokenizer may not have correct special tokens set.
    # Check and add them if missing to prevent them from being parsed into different tokens.
    # Note that these are present in the vocabulary.
    # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
    logger.info('Adding special tokens.')
    if 'llama' in args.model_name_or_path or 'baichuan' in args.model_name_or_path:
        add_special_tokens_if_missing(tokenizer, model)

    return model, tokenizer
