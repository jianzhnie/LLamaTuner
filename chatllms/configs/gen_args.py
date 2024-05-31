from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GenerationArguments:
    """Arguments pertaining to specify the model generation parameters."""

    # Generation strategy
    # 是否采样
    do_sample: Optional[bool] = field(
        default=True,
        metadata={
            'help':
            'Whether or not to use sampling, use greedy decoding otherwise.'
        },
    )
    # Hyperparameters for logit manipulation
    # softmax 函数的温度因子，来调节输出token的分布
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            'help': 'The value used to modulate the next token probabilities.'
        },
    )
    # 核采样参数，top_p最高的前n个（n是变化）概率和为p，从这些n个候选token中随机采样
    top_p: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
            'The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept.'
        },
    )
    # top_k随机搜索中的k个最高概率选择
    top_k: Optional[int] = field(
        default=50,
        metadata={
            'help':
            'The number of highest probability vocabulary tokens to keep for top-k filtering.'
        },
    )
    # 集束搜索的数量
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            'help': 'Number of beams for beam search. 1 means no beam search.'
        },
    )
    # 最大的token数量，会被 max_new_tokens 覆盖
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            'help':
            'The maximum length the generated tokens can have. It can be overridden by max_new_tokens.'
        },
    )
    # 最大的新生成的token数量
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={
            'help':
            'Maximum number of new tokens to be generated in evaluation or prediction loops'
            'if predict_with_generate is set.'
        },
    )
    # 重复性惩罚因子
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
            'The parameter for repetition penalty. 1.0 means no penalty.'
        })
    # 长度惩罚因子
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
            'Exponential penalty to the length that is used with beam-based generation.'
        })
    default_system: Optional[str] = field(
        default=None,
        metadata={'help': 'Default system message to use in chat completion.'},
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get('max_new_tokens', None):
            args.pop('max_length', None)
        else:
            args.pop('max_new_tokens', None)
        return args
