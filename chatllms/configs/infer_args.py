from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelInferenceArguments:
    cache_dir: Optional[str] = field(default=None)
    full_finetune: bool = field(
        default=False,
        metadata={'help': 'Finetune the entire model without adapters.'})
    gradient_checkpointing: bool = field(
        default=True,
        metadata={'help': 'Use gradient checkpointing. You want to use this.'})
    model_name_or_path: Optional[str] = field(
        default='facebook/opt-125m',
        metadata={'help': 'Path to pre-trained model'})
    checkpoint_dir: Optional[str] = field(
        default=None, metadata={'help': 'Path to pre-trained lora model'})
    model_max_length: int = field(
        default=2048,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    prompt_template: Optional[str] = field(
        default='default',
        metadata={
            'help':
            'Which template to use for constructing prompts in training and inference.'
        })
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'A prefix to add before every source text. Use `|` to separate multiple prefixes.'
        })
    double_quant: bool = field(
        default=True,
        metadata={
            'help':
            'Compress the quantization statistics through double quantization.'
        })
    quant_type: str = field(
        default='nf4',
        metadata={
            'help':
            'Quantization data type to use. Should be one of `fp4` or `nf4`.'
        })
    bits: int = field(default=4, metadata={'help': 'How many bits to use.'})
    fp16: bool = field(default=False, metadata={'help': 'Use fp16.'})
    bf16: bool = field(default=False, metadata={'help': 'Use bf16.'})
    max_memory_MB: int = field(default=8000,
                               metadata={'help': 'Free memory per gpu.'})
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
