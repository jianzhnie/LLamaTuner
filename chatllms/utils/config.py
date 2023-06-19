from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')
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


@dataclass
class DataArguments:
    dataset_name: str = field(
        default='alpaca',
        metadata={
            'help': 'Which dataset to finetune on. See datamodule for options.'
        })
    data_dir: str = field(
        default='./data',
        metadata={
            'help':
            'where is dataset in local dir. See datamodule for options.'
        })
    load_from_local: bool = field(
        default=False,
        metadata={
            'help': 'To load the data from local or  huggingface data hub?'
        })
    eval_dataset_size: Optional[float] = field(
        default=0.1, metadata={'help': 'Size of validation dataset.'})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'For debugging purposes or quicker training, truncate the number of training examples to this '
            'value if set.'
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'For debugging purposes or quicker training, truncate the number of evaluation examples to this '
            'value if set.'
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={
            'help':
            'Maximum source sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    target_max_len: int = field(
        default=256,
        metadata={
            'help':
            'Maximum target sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    full_finetune: bool = field(
        default=False,
        metadata={'help': 'Finetune the entire model without adapters.'})
    do_train: bool = field(
        default=False,
        metadata={'help': 'To train or not to train, that is the question?'})
    do_eval: bool = field(
        default=False,
        metadata={'help': 'To train or not to train, that is the question?'})
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Whether to train on the input in addition to the target text.'
        })
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to run the MMLU evaluation.'})
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={
            'help':
            'MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot.'
        })
    mmlu_split: Optional[str] = field(
        default='eval', metadata={'help': 'The MMLU split to run on'})
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset.'
        })
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={'help': 'Maximum source sequence length for mmlu.'})
    sample_generate: bool = field(
        default=False,
        metadata={'help': 'If do sample generation on evaluation.'})
    optim: str = field(default='paged_adamw_32bit',
                       metadata={'help': 'The optimizer to be used'})
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            'help':
            'Gradient clipping max norm. This is tuned and works well for all models tested.'
        })
    gradient_checkpointing: bool = field(
        default=True,
        metadata={'help': 'Use gradient checkpointing. You want to use this.'})
    predict_with_generate: bool = field(
        default=False,
        metadata={
            'help':
            'Group sequences into batches with same length. Saves memory and speeds up training considerably.'
        })


@dataclass
class LoraArguments:
    lora_r: int = field(default=64, metadata={'help': 'Lora R dimension.'})
    lora_alpha: float = field(default=16, metadata={'help': ' Lora alpha.'})
    lora_dropout: float = field(default=0.0,
                                metadata={'help': 'Lora dropout.'})
    max_memory_MB: int = field(default=80000,
                               metadata={'help': 'Free memory per gpu.'})
    lora_weight_path: str = ''
    bias: str = 'none'


@dataclass
class QuantArgments:
    adam8bit: bool = field(default=False, metadata={'help': 'Use 8-bit adam.'})
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


@dataclass
class GenerationArguments:
    # generation parameters
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            'help':
            'Maximum number of new tokens to be generated in evaluation or prediction loops'
            'if predict_with_generate is set.'
        })
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'Minimum number of new tokens to generate.'})

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)
