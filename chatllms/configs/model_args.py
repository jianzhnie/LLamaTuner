from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default='facebook/opt-125m',
        metadata={
            'help':
            ("The model checkpoint for weights initialization. Don't set if you want to\
              train a model from scratch.")
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained tokenizer name or path if not the same as model_name'
        })
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
    padding_side: str = field(
        default='right', metadata={'help': 'The padding side in tokenizer'})
