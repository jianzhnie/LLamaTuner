from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import transformers


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
    multiturn_dialogue: Optional[str] = field(
        default=False,
        metadata={
            'help': 'The dataset is a multiturn_dialogue dataset or not'
        })
    lazy_preprocess: bool = True
    # 微调数据集是alpaca，那么可以试试中文的效果。Llama、Bloom和OPT，或者MPT等等
    dataset_name: Optional[str] = field(
        default='alpaca',
        metadata={
            'help': 'Which dataset to finetune on. See datamodule for options.'
        })
    data_path: str = field(
        default='./data',
        metadata={
            'help':
            'where is dataset in local dir. See datamodule for options.'
        })
    # 数据集的本地路径，如果load_from_local为True，那么就从本地加载数据集
    data_dir: str = field(
        default='./data',
        metadata={
            'help':
            'where is dataset in local dir. See datamodule for options.'
        })
    # 是否从本地加载数据集
    load_from_local: bool = field(
        default=False,
        metadata={
            'help': 'To load the data from local or  huggingface data hub?'
        })
    # 验证数据集的尺寸，也就是数量
    eval_dataset_size: Optional[float] = field(
        default=0.1, metadata={'help': 'Size of validation dataset.'})
    # 最大训练数据样本的数量。主要是为了快速调试训练代码
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'For debugging purposes or quicker training, truncate the number of training examples to this '
            'value if set.'
        },
    )
    # 与max_train_samples类似，主要是为了快速调试训练代码
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'For debugging purposes or quicker training, truncate the number of evaluation examples to this '
            'value if set.'
        },
    )
    # 最大文本输入的最大长度。如果source文本token长度超过该值，需要做文本的截断
    source_max_len: int = field(
        default=1024,
        metadata={
            'help':
            'Maximum source sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    # 标签文本的最大长度，如果target文本token长度超过该值，需要做文本的截断
    target_max_len: int = field(
        default=256,
        metadata={
            'help':
            'Maximum target sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # 缓存目录
    cache_dir: Optional[str] = field(default=None)
    # 不使用adapter进行全微调（不适用Lora或qlora？）
    full_finetune: bool = field(
        default=False,
        metadata={'help': 'Finetune the entire model without adapters.'})
    # 是否进行训练，那肯定是要的
    do_train: bool = field(
        default=False,
        metadata={'help': 'To train or not to train, that is the question?'})
    # 是否进行验证
    do_eval: bool = field(
        default=False,
        metadata={'help': 'To train or not to train, that is the question?'})
    # 是否在source文本上进行GPT LM微调。默认是False，这部分文本对应的token会在label中设置为-100的标签
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Whether to train on the input in addition to the target text.'
        })
    # 是否使用MMLU评估
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to run the MMLU evaluation.'})
    # mmlu数据集的默认名称，`mmlu-zs` for zero-shot or `mmlu-fs` for few shot.
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={
            'help':
            'MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot.'
        })
    # mmlu数据集的默认分割，`eval` for evaluation or `test` for testing.
    mmlu_split: Optional[str] = field(
        default='eval', metadata={'help': 'The MMLU split to run on'})
    # mmlu数据集的默认最大样本数量
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset.'
        })
    # mmlu数据集source文本的最大长度（是字符长度还是token长度，这个去代码中找线索吧）
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={'help': 'Maximum source sequence length for mmlu.'})
    # 是否进行sample generation
    sample_generate: bool = field(
        default=False,
        metadata={'help': 'If do sample generation on evaluation.'})
    # 使用nvidia的分页机制优化器，可以在偶尔OOM的情况，让模型继续训练下去。
    optim: str = field(default='paged_adamw_32bit',
                       metadata={'help': 'The optimizer to be used'})
    # 梯度截断因子
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            'help':
            'Gradient clipping max norm. This is tuned and works well for all models tested.'
        })
    # 梯度检查，设置为True，来减少显存占用。
    # 显存这么紧张，肯定是要设置为 True，但是运行时间就会提升
    gradient_checkpointing: bool = field(
        default=True,
        metadata={'help': 'Use gradient checkpointing. You want to use this.'})
    predict_with_generate: bool = field(
        default=False,
        metadata={
            'help':
            'Group sequences into batches with same length. Saves memory and speeds up training considerably.'
        })
    model_max_length: int = field(
        default=2048,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


@dataclass
class LoraArguments:
    # lora中A矩阵的列数量和B矩阵的行数量
    lora_r: int = field(default=64, metadata={'help': 'Lora R dimension.'})
    # 缩放因子
    lora_alpha: float = field(default=16, metadata={'help': ' Lora alpha.'})
    #  dropout，一种正则化方法，可以模仿集成学习
    lora_dropout: float = field(default=0.0,
                                metadata={'help': 'Lora dropout.'})
    # 每个GPU上可使用的显存大小，以MB为单位。默认是A100高端版本的80GB
    max_memory_MB: int = field(default=8000,
                               metadata={'help': 'Free memory per gpu.'})
    lora_weight_path: str = ''
    bias: str = 'none'


@dataclass
class QuantArgments:
    # 使用8-bit的adam，是否可以调整为LION或Sophia，甚至deepspeed还提供了多个1-bit优化器选择
    adam8bit: bool = field(default=False, metadata={'help': 'Use 8-bit adam.'})
    # 是否使用二次量化
    double_quant: bool = field(
        default=True,
        metadata={
            'help':
            'Compress the quantization statistics through double quantization.'
        })
    # 量化类型，可以选择`fp4`或`nf4`
    quant_type: str = field(
        default='nf4',
        metadata={
            'help':
            'Quantization data type to use. Should be one of `fp4` or `nf4`.'
        })
    # 使用的位宽，默认为4。
    bits: int = field(default=4, metadata={'help': 'How many bits to use.'})

    def __post_init__(self):
        if self.bits is not None:
            assert self.bits in [
                4, 8
            ], 'We only accept 4-bit or 8-bit quantization.'

        if self.quant_type is not None:
            assert self.quant_type in [
                'nf4', 'fp4'
            ], 'We only accept `nf4` or `fp4` quantization type.'


@dataclass
class GenerationArguments:
    # generation parameters
    # Length arguments
    # 最大的新生成的token数量
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={
            'help':
            'Maximum number of new tokens to be generated in evaluation or prediction loops'
            'if predict_with_generate is set.'
        })
    # 最少的新生成的token数量
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'Minimum number of new tokens to generate.'})
    # 最大的token数量，会被 max_new_tokens 覆盖
    max_length: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The maximum length the generated tokens can have. It can be overridden by max_new_tokens.'
        })
    # Generation strategy
    # 是否采样
    do_sample: Optional[bool] = field(default=True)
    # 集束搜索的数量
    num_beams: Optional[int] = field(default=1)
    # 集束搜索的组数量
    num_beam_groups: Optional[int] = field(default=1)
    # 惩罚因子
    penalty_alpha: Optional[float] = field(default=None)
    # 是否使用cache
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    # softmax 函数的温度因子，来调节输出token的分布
    temperature: Optional[float] = field(default=1.0)
    # top_k随机搜索中的k个最高概率选择
    top_k: Optional[int] = field(default=50)
    # 核采样参数，top_p最高的前n个（n是变化）概率和为p，从这些n个候选token中随机采样
    top_p: Optional[float] = field(default=1.0)
    # 典型p值
    typical_p: Optional[float] = field(default=1.0)
    # 丰富性惩罚因子
    diversity_penalty: Optional[float] = field(default=0.0)
    # 重复性惩罚因子
    repetition_penalty: Optional[float] = field(default=1.0)
    # 长度惩罚因子
    length_penalty: Optional[float] = field(default=1.0)
    # 没有ngram重复的尺度大小
    # 一般随机采样的丰富性够了，所以一般不会设置，如果重复很多则设置为2是比较好的选择
    no_repeat_ngram_size: Optional[int] = field(default=0)

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get('max_new_tokens', None):
            args.pop('max_length', None)
        return args
