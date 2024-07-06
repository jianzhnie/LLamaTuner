from typing import List

import bitsandbytes as bnb
import torch
from robin.LLamaTuner.llamatuner.configs.finetuning_args import \
    FinetuningArguments
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

from llamatuner.utils.logger_utils import get_logger

logger = get_logger('llamatuner')


def find_all_linear_names(finetune_args: FinetuningArguments,
                          model: torch.nn.Module) -> List[str]:
    """Returns a list of names of all linear layers present in the given model.
    如果args.bits是4，使用bitsandbytes库中的bnb.nn.Linear4bit层；
    如果args.bits是8，使用bitsandbytes库中的bnb.nn.Linear8bitLt层； 否则，使用torch.nn.Linear层；
    并记录下这些层的名称，保存在lora_module_names集合中。

    Args:
        args (argparse.Namespace): A namespace containing arguments of the script.
        model (torch.nn.Module): The PyTorch model to extract linear layer names from.

    Returns:
        List[str]: A list of names of all linear layers present in the given model.

    Raises:
        TypeError: If `args` is not an instance of `argparse.Namespace`, or if `model` is not an instance \
            of `torch.nn.Module`.
        ValueError: If `args.bits` is not 4 or 8.

    Example Usage:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--bits', type=int)
        >>> args = parser.parse_args(['--bits', '4'])
        >>> model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 1))
        >>> find_all_linear_names(args, model)
        ['0', '1']
    """
    # Determine the correct linear layer class based on the value of `args.bits`
    if finetune_args.quant_bit == 4:
        cls = bnb.nn.Linear4bit
    elif finetune_args.quant_bit == 8:
        cls = bnb.nn.Linear8bitLt
    else:
        cls = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        # Check if the current module is an instance of the linear layer class
        if isinstance(module, cls):
            # If yes, split the name of the module into its component parts and add the first or last part to the set
            names = name.split('.')
            # 只保留最后的名称，前缀不保留
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Remove 'lm_head' from the set if present (needed for 16-bit)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    # Convert the set into a list and return it
    return list(lora_module_names)


def find_all_linear_modules(model: PreTrainedModel,
                            freeze_vision_tower: bool) -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    forbidden_modules = {'lm_head'}

    if model.config.model_type == 'chatglm':
        forbidden_modules.add('output_layer')
    elif model.config.model_type == 'internlm2':
        forbidden_modules.add('output')
    elif model.config.model_type in ['llava', 'paligemma']:
        forbidden_modules.add('multi_modal_projector')

    if freeze_vision_tower:
        forbidden_modules.add('vision_tower')

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name
               for forbidden_module in forbidden_modules):
            continue

        if ('Linear' in module.__class__.__name__
                and 'Embedding' not in module.__class__.__name__):
            module_names.add(name.split('.')[-1])

    logger.info('Found linear modules: %s', ','.join(module_names))
    return list(module_names)


def find_expanded_modules(model: 'PreTrainedModel', target_modules: List[str],
                          num_layer_trainable: int) -> List[str]:
    r"""
    Finds the modules in the expanded blocks to apply lora.
    """
    num_layers = getattr(model.config, 'num_hidden_layers', None)
    if not num_layers:
        raise ValueError('Model was not supported.')

    if num_layers % num_layer_trainable != 0:
        raise ValueError(
            '`num_layers` {} should be divisible by `num_layer_trainable` {}.'.
            format(num_layers, num_layer_trainable))

    stride = num_layers // num_layer_trainable
    trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    trainable_layers = ['.{:d}.'.format(idx) for idx in trainable_layer_ids]
    module_names = []
    for name, _ in model.named_modules():
        if any(target_module in name
               for target_module in target_modules) and any(
                   trainable_layer in name
                   for trainable_layer in trainable_layers):
            module_names.append(name)

    logger.info('Apply lora to layers: %s',
                ','.join(map(str, trainable_layer_ids)))
    return module_names


def register_autoclass(
    config: 'PretrainedConfig',
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer',
):
    if 'AutoConfig' in getattr(config, 'auto_map', {}):
        config.__class__.register_for_auto_class()
    if 'AutoModelForCausalLM' in getattr(config, 'auto_map', {}):
        model.__class__.register_for_auto_class()
    if 'AutoTokenizer' in tokenizer.init_kwargs.get('auto_map', {}):
        tokenizer.__class__.register_for_auto_class()
