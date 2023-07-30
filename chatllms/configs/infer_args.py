from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelInferenceArguments:
    cache_dir: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(
        default='facebook/opt-125m',
        metadata={'help': 'Path to pre-trained model'})
    model_max_length: int = field(
        default=2048,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
