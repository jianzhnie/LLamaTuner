from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer."""

    model_name_or_path: Optional[str] = field(
        default='facebook/opt-125m',
        metadata={
            'help':
            ('Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models.'
             )
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            'help':
            ('Path to the adapter weight or identifier from huggingface.co/models. '
             'Use commas to separate multiple adapters.')
        },
    )
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The folder containing the adapter weights to load.'
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            'help':
            'Whether or not to use one of the fast tokenizer (backed by the tokenizers library).'
        },
    )
    resize_vocab: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to resize the tokenizer vocab and the embedding layers.'
        },
    )
    model_max_length: Optional[int] = field(
        default=1024,
        metadata={
            'help':
            'The maximum length of the model input, including special tokens.'
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={
            'help':
            'Whether or not to trust the remote code in the model configuration.'
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn.'
        },
    )
    model_revision: str = field(
        default='main',
        metadata={
            'help':
            'The specific model version to use (can be a branch name, tag name or commit id).'
        },
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not the special tokens should be split during the tokenization process.'
        },
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={'help': 'Special tokens to be added into the tokenizer.'},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            'help': 'Whether or not to use memory-efficient model loading.'
        },
    )
    rope_scaling: Optional[Literal['linear', 'dynamic']] = field(
        default=None,
        metadata={
            'help':
            'Which scaling strategy should be adopted for the RoPE embeddings.'
        },
    )
    flash_attn: Literal['off', 'sdpa', 'fa2', 'auto'] = field(
        default='auto',
        metadata={
            'help': 'Enable FlashAttention for faster training and inference.'
        },
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={
            'help': 'Whether or not to randomly initialize the model weights.'
        },
    )
    offload_folder: str = field(
        default='offload',
        metadata={'help': 'Path to offload model weights.'},
    )
    use_cache: bool = field(
        default=True,
        metadata={'help': 'Whether or not to use KV cache in generation.'},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={'help': 'Auth token to log in with Hugging Face Hub.'},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={'help': 'Auth token to log in with ModelScope Hub.'},
    )

    def __post_init__(self):
        self.compute_dtype = None
        self.device_map = None

        if self.model_name_or_path is None:
            raise ValueError('Please provide `model_name_or_path`.')

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [
                path.strip() for path in self.adapter_name_or_path.split(',')
            ]

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError(
                '`split_special_tokens` is only supported for slow tokenizers.'
            )

        if self.new_special_tokens is not None:  # support multiple special tokens
            self.new_special_tokens = [
                token.strip() for token in self.new_special_tokens.split(',')
            ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
