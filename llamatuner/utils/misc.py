import gc
import os
from typing import Tuple

import torch
from transformers.utils import (is_torch_bf16_gpu_available,
                                is_torch_cuda_available,
                                is_torch_mps_available, is_torch_npu_available,
                                is_torch_xpu_available)
from transformers.utils.versions import require_version

from llamatuner.utils.logger_utils import get_logger

_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except Exception:
    _is_bf16_available = False

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
        logger.warning(
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


def get_peak_memory() -> Tuple[int, int]:
    r"""
    Gets the peak memory usage for the current device (in Bytes).
    """
    if is_torch_npu_available():
        return torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved(
        )
    elif is_torch_cuda_available():
        return torch.cuda.max_memory_allocated(
        ), torch.cuda.max_memory_reserved()
    else:
        return 0, 0


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
    Collects GPU or NPU memory.
    """
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


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
