from dataclasses import dataclass, field


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
    max_memory_MB: int = field(default=80000,
                               metadata={'help': 'Free memory per gpu.'})
    lora_weight_path: str = ''
    bias: str = 'none'
