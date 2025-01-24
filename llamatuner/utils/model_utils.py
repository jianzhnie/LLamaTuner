import argparse
import os
from os.path import exists, isdir, join
from typing import Any, Dict, List, Tuple, Union

import bitsandbytes as bnb
import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from transformers.trainer_utils import get_last_checkpoint

from llamatuner.configs.finetuning_args import FinetuningArguments
from llamatuner.utils.constants import (DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN,
                                        DEFAULT_PAD_TOKEN, DEFAULT_UNK_TOKEN)
from llamatuner.utils.logger_utils import get_logger

logger = get_logger(__name__)


def find_all_linear_names(
    model: torch.nn.Module,
    finetuning_args: FinetuningArguments,
) -> List[str]:
    """Returns a list of names of all linear layers present in the given model.
    如果args.bits是4，使用bitsandbytes库中的bnb.nn.Linear4bit层；
    如果args.bits是8，使用bitsandbytes库中的bnb.nn.Linear8bitLt层； 否则，使用torch.nn.Linear层；
    并记录下这些层的名称，保存在lora_module_names集合中。

    Args:
        model (torch.nn.Module): The PyTorch model to extract linear layer names from.
        finetuning_args (argparse.Namespace): A namespace containing arguments of the script.

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
    if finetuning_args.quant_bit == 4:
        cls = bnb.nn.Linear4bit
    elif finetuning_args.quant_bit == 8:
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


def find_all_linear_modules(model: PreTrainedModel) -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    forbidden_modules = {'lm_head'}

    if model.config.model_type == 'chatglm':
        forbidden_modules.add('output_layer')
    elif model.config.model_type == 'internlm2':
        forbidden_modules.add('output')

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


def add_special_tokens_if_missing(tokenizer: PreTrainedTokenizer,
                                  model: PreTrainedModel) -> None:
    """If 'llama' or 'baichuan' is in the model name or path, check if the
    special tokens are set correctly. Add any missing special tokens to prevent
    them from being parsed into different tokens. Note that these special
    tokens are present in the vocabulary. Note also that
    `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.

    Args:
        tokenizer: The pre-trained tokenizer.
        model: The pre-trained model.

    Returns:
        None.
    """
    # Define a dictionary to store any missing special tokens along with their default values
    special_tokens_dict: Dict[str, Any] = {}

    # Check if each special token is present. If not, add it to the special_tokens_dict with its default value.
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    # If there are any missing special tokens, call `smart_tokenizer_and_embedding_resize()` to add them to the
    # tokenizer and resize the embedding accordingly.
    if len(special_tokens_dict) > 0:
        smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer,
                                             model)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> None:
    """Resize tokenizer and embedding to accommodate new special tokens.
    改变tokenizer和embedding的尺寸。 一般需要将tokenizer和embedding的尺寸设置为64的倍数，方便GPU加速。

    Args:
        special_tokens_dict (Dict[str, str]): A dictionary of special tokens to be added to the tokenizer.
        tokenizer (PreTrainedTokenizer): The tokenizer object to be resized.
        model (PreTrainedModel): The model object whose token embeddings are to be resized.

    Returns:
        None

    Note: This function resizes the tokenizer to accommodate additional special tokens and the
    embedding matrix of the model to match the new size of the tokenizer. If any new special tokens
    have been added, the function computes the average embedding values of the existing embeddings
    and sets those values for the new special token embeddings. This is done separately for the input
    embeddings and output embeddings of the model.
    """
    # 添加特殊token字典，并且得到新加入字典的token数量
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    # Resize token embeddings to match tokenizer
    # 更改token_embeddings的尺寸
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        # Compute average embeddings of existing tokens
        # 下面的操作实现使用已训练好的embedding的均值，来初始化新token对应的embedding
        # input_embeddings的已训练好的embedding的均值，保持embedding的shape
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        # output_embeddings的已训练好的embedding的均值，保持embedding的shape
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        # Set average embeddings for new special token embeddings
        # 分别给input_embeddings和output_embeddings的新token对应的embedding赋值
        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def print_trainable_parameters(model: torch.nn.Module, kbit: int = 8) -> None:
    """Prints the number of trainable parameters in the given model.

    Args:
        model (torch.nn.Module): The PyTorch model to count trainable parameters in.

    Raises:
        TypeError: If `args` is not an instance of `argparse.Namespace`, or if `model` is not an instance \
            of `torch.nn.Module`.

    Example Usage:
        >>> model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 1))
        >>> print_trainable_parameters(model, kbit=4)
        trainable params: 13.0 || all params: 61 || trainable: 21.311475409836067%
    """
    trainable_params = 0
    all_param = 0

    # Iterate through all the named parameters of the model
    for _, param in model.named_parameters():
        all_param += param.numel()
        # Add the number of elements in the parameter tensor to the total count
        if (
                param.requires_grad
        ):  # If the parameter requires gradient computation during backpropagation
            trainable_params += param.numel()
            # Add its number of elements to the trainable parameters count

    # If args.bits is 4, divide the trainable params count by 2 \
    # (since each 4-bit element requires only 2 bits for storage)
    if kbit == 4:
        trainable_params /= 2

    # Compute and print the percentage of trainable vs all parameters
    trainable_percent = 100 * trainable_params / all_param
    logger.info(f'trainable params: {trainable_params} || '
                f'all params: {all_param} || '
                f'trainable: {trainable_percent}%')


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, 'ds_numel'):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == 'Params4bit':
            if hasattr(param, 'quant_storage') and hasattr(
                    param.quant_storage, 'itemsize'):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, 'element_size'):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_model_dtypes(model: torch.nn.Module) -> None:
    """检查模型参数的数据类型，并输出各个数据类型在这些张量中所占的比例.

    :param model: 待检查的模型.
    :return: 无返回值.
    """
    # 创建一个空字典dtypes，用于记录模型参数的数据类型及其数量.
    dtypes = {}

    # 遍历模型参数，并统计每种数据类型出现的次数.
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()

    # 计算总共的参数数量total.
    total = sum(dtypes.values())

    # 输出各个数据类型的数量以及所占比例.
    for k, v in dtypes.items():
        logger.info(f'{k}: {v} ({100 * v / total:.2f}%)')
    return None


def check_training_finished(args: argparse.Namespace) -> Tuple[str, bool]:
    """Given a directory containing previous saved checkpoints, returns the
    path to the last checkpoint if available along with a boolean flag
    indicating whether training has already been completed.

    Args:
        checkpoint_dir (str): Path to the directory containing the saved checkpoints.

    Returns:
        A tuple containing the path to the last checkpoint if available, and a boolean flag indicating
        whether training has already been completed.
    """
    # Check if provided directory exists
    if isdir(args.output_dir) and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint:
            logger.info(
                f'Find lasest checkpoint: ({last_checkpoint}) in ({args.output_dir})'
            )
        # Check if 'completed' file exists in the directory - indicates training has completed
        is_completed = exists(join(args.output_dir, 'completed'))
        if last_checkpoint and is_completed:
            raise AssertionError(
                f'Detected that training was already completed! Output directory ({args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.')

        elif last_checkpoint:
            # Return path to the latest checkpoint directory
            logger.info(
                f'Checkpoint detected, resuming training at ({last_checkpoint}). To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
            return last_checkpoint, is_completed
    # The directory does not exist, meaning this is the first time the training is being run
    logger.info(
        f'The output directory: ({args.output_dir}) do not exists or emppty or you have set --overwrite_output_dir... will train from scratch'
    )
    return None, False  # first training


def find_last_checkpoint(checkpoint_dir):
    # Find the latest checkpoint by checking all subdirectories named 'checkpoint-*'
    max_step = 0
    last_checkpoint = None
    for filename in os.listdir(checkpoint_dir):
        if isdir(join(checkpoint_dir,
                      filename)) and filename.startswith('checkpoint'):
            max_step = max(max_step, int(filename.replace('checkpoint-', '')))
    if max_step > 0:
        last_checkpoint = join(checkpoint_dir, f'checkpoint-{max_step}')
    return last_checkpoint


# Avoid runtime error in model.generate(do_sample=True).
class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 0] = 1.0
        return scores


def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    return logits_processor


def trainer_save_model_safe(trainer: Trainer, output_dir: str = None):
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT,
                              save_policy):
        trainer.save_model(output_dir)


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model(output_dir)
    else:
        trainer_save_model_safe(trainer)


def maybe_zero_3(param: Union[torch.Tensor, object]) -> torch.Tensor:
    """Applies zero.GatheredParameters to gather the parameter if it has ds_id
    attribute, and clones and detaches the tensor data if ds_status is
    ZeroParamStatus.NOT_AVAILABLE.

    Args:
        param: The parameter to be processed.

    Returns:
        The modified parameter.

    Raises:
        AssertionError: If `param` has ds_id attribute but ds_status is not ZeroParamStatus.NOT_AVAILABLE.
    """
    if hasattr(param, 'ds_id'):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE, 'Invalid ds_status'

        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params: List[Tuple[str, torch.Tensor]],
                                bias: str) -> Dict[str, torch.Tensor]:
    """Filters and processes named parameters based on the specified bias.

    Args:
        named_params: An iterable containing tuples of parameter names and their corresponding values.
        bias: The bias type.

    Returns:
        A dictionary containing the filtered and possibly modified named parameters.

    Raises:
        NotImplementedError: If an unsupported bias type is provided.
    """
    to_return: Dict[str, torch.Tensor] = {}

    if bias == 'none':
        to_return = {k: t for k, t in named_params if 'lora_' in k}
    elif bias == 'all':
        to_return = {
            k: t
            for k, t in named_params if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        maybe_lora_bias: Dict[str, torch.Tensor] = {}
        lora_bias_names: set() = set()

        for k, t in named_params:
            if 'lora_' in k:
                to_return[k] = t
                bias_name = k.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in k:
                maybe_lora_bias[k] = t

        for k, t in maybe_lora_bias.items():
            bias_name = k.split('bias')[0] + 'bias'
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError('Unsupported bias type')

    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}

    return to_return
