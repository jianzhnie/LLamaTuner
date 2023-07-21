from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


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
