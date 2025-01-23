import gc
import os
from typing import Dict, Tuple

import torch
from peft import PeftModel
from transformers import (InfNanRemoveLogitsProcessor, LogitsProcessorList,
                          PreTrainedModel)
from transformers.utils import (SAFE_WEIGHTS_NAME, WEIGHTS_NAME,
                                is_torch_bf16_gpu_available,
                                is_torch_cuda_available,
                                is_torch_mps_available, is_torch_npu_available,
                                is_torch_xpu_available)
from transformers.utils.versions import require_version

from llamatuner.utils.constants import (V_HEAD_SAFE_WEIGHTS_NAME,
                                        V_HEAD_WEIGHTS_NAME)
from llamatuner.utils.logger_utils import get_logger

_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except Exception:
    _is_bf16_available = False

from trl import AutoModelForCausalLMWithValueHead

from llamatuner.configs.model_args import ModelArguments

logger = get_logger('llamatuner')


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_version(requirement: str, mandatory: bool = False) -> None:
    r"""
    Optionally checks the package version.
    """
    if os.getenv('DISABLE_VERSION_CHECK', '0').lower() in ['true', '1'
                                                           ] and not mandatory:
        logger.warning_rank0_once(
            'Version checking has been disabled, may lead to unexpected behaviors.'
        )
        return

    if mandatory:
        hint = f'To fix: run `pip install {requirement}`.'
    else:
        hint = f'To fix: run `pip install {requirement}` or set `DISABLE_VERSION_CHECK=1` to skip this check.'

    require_version(requirement, hint)


def check_dependencies() -> None:
    r"""
    Checks the version of the required packages.
    """
    check_version('transformers>=4.41.2,<=4.46.1')
    check_version('datasets>=2.16.0,<=3.1.0')
    check_version('accelerate>=0.34.0,<=1.0.1')
    check_version('peft>=0.11.1,<=0.12.0')
    check_version('trl>=0.8.6,<=0.9.6')


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


def fix_valuehead_checkpoint(
    model: 'AutoModelForCausalLMWithValueHead',
    output_dir: str,
    safe_serialization: bool,
) -> None:
    r"""
    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        from safetensors import safe_open
        from safetensors.torch import save_file

        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework='pt', device='cpu') as f:
            state_dict: Dict[str, torch.Tensor] = {
                key: f.get_tensor(key)
                for key in f.keys()
            }
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: Dict[str, torch.Tensor] = torch.load(path_to_checkpoint,
                                                         map_location='cpu')

    decoder_state_dict = {}
    v_head_state_dict = {}
    for name, param in state_dict.items():
        if name.startswith('v_head.'):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace('pretrained_model.', '')] = param

    os.remove(path_to_checkpoint)
    model.pretrained_model.save_pretrained(
        output_dir,
        state_dict=decoder_state_dict or None,
        safe_serialization=safe_serialization,
    )

    if safe_serialization:
        save_file(
            v_head_state_dict,
            os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            metadata={'format': 'pt'},
        )
    else:
        torch.save(v_head_state_dict,
                   os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info('Value head model saved at: {}'.format(output_dir))


def get_current_device() -> torch.device:
    r"""
    Gets the current available device.
    """
    if is_torch_xpu_available():
        device = 'xpu:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_npu_available():
        device = 'npu:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_mps_available():
        device = 'mps:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    else:
        device = 'cpu'

    return torch.device(device)


def get_device_count() -> int:
    r"""
    Gets the number of available GPU or NPU devices.
    """
    if is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def get_logits_processor() -> 'LogitsProcessorList':
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def is_gpu_or_npu_available() -> bool:
    r"""
    Checks if the GPU or NPU is available.
    """
    return is_torch_npu_available() or is_torch_cuda_available()


def has_tokenized_data(path: os.PathLike) -> bool:
    r"""
    Checks if the path has a tokenized dataset.
    """
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def try_download_model_from_ms(model_args: 'ModelArguments') -> str:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    try:
        from modelscope import snapshot_download

        revision = ('master' if model_args.model_revision == 'main' else
                    model_args.model_revision)
        return snapshot_download(
            model_args.model_name_or_path,
            revision=revision,
            cache_dir=model_args.cache_dir,
        )
    except ImportError as exc:
        raise ImportError(
            'Please install modelscope via `pip install modelscope -U`'
        ) from exc


def use_modelscope() -> bool:
    return os.environ.get('USE_MODELSCOPE_HUB', '0').lower() in ['true', '1']
