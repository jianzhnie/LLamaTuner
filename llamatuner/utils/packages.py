import importlib.metadata
import importlib.util
from typing import TYPE_CHECKING

from packaging import version

if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> 'Version':
    try:
        return version.parse(importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        return version.parse('0.0.0')


def is_fastapi_available():
    return _is_package_available('fastapi')


def is_gradio_available():
    return _is_package_available('gradio')


def is_jieba_available():
    return _is_package_available('jieba')


def is_matplotlib_available():
    return _is_package_available('matplotlib')


def is_nltk_available():
    return _is_package_available('nltk')


def is_pillow_available():
    return _is_package_available('PIL')


def is_requests_available():
    return _is_package_available('requests')


def is_rouge_available():
    return _is_package_available('rouge_chinese')


def is_starlette_available():
    return _is_package_available('sse_starlette')


def is_uvicorn_available():
    return _is_package_available('uvicorn')


def is_vllm_available():
    return _is_package_available('vllm')
