from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelInferenceArguments:
    cache_dir: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(
        default='facebook/opt-125m',
        metadata={'help': 'Path to pre-trained model'})
    model_revision: str = field(
        default='main',
        metadata={
            'help':
            'The specific model version to use (can be a branch name, tag name or commit id).'
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained.'
        })
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Enables using Huggingface auth token from Git Credentials.'
        })
    model_max_length: int = field(
        default=2048,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={'help': 'Whether to use low cpu memory usage mode.'})
    fp16: bool = field(default=False,
                       metadata={'help': 'Whether to use fp16.'})
    prompt_template: str = field(
        default='default',
        metadata={
            'help':
            'Prompt template name. Such as vanilla, alpaca, llama2, vicuna..., etc.'
        })
    source_prefix: Optional[str] = field(
        default=None,
        metadata={'help': 'Prefix to prepend to every source text.'})
