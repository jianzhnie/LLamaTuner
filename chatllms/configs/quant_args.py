from dataclasses import dataclass, field


@dataclass
class QuantArguments:
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
