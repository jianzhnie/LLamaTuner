from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class QuantArguments:
    # 使用8-bit的adam，是否可以调整为LION或Sophia，甚至deepspeed还提供了多个1-bit优化器选择
    adam8bit: bool = field(default=False, metadata={'help': 'Use 8-bit adam.'})
    # 使用的位宽，默认为4。
    quant_bit: Optional[int] = field(
        default=4,
        metadata={
            'help':
            'The number of bits to quantize the model using bitsandbytes.'
        },
    )
    # 量化类型，可以选择`fp4`或`nf4`
    quant_type: Literal['fp4', 'nf4'] = field(
        default='nf4',
        metadata={
            'help': 'Quant data type to use. Should be one of `fp4` or `nf4`.'
        },
    )
    # 是否使用二次量化
    double_quant: bool = field(
        default=True,
        metadata={
            'help': 'Compress the quant statistics through double quant.'
        },
    )
    quant_device_map: Optional[Literal['auto']] = field(
        default=None,
        metadata={
            'help':
            'Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0.'
        },
    )
    export_quant_bit: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of bits to quantize the exported model.'
        },
    )
    export_quant_dataset: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Path to the dataset or dataset name to use in quantizing the exported model.'
        },
    )
    export_quant_nsamples: int = field(
        default=128,
        metadata={'help': 'The number of samples used for quant.'},
    )
    export_quant_maxlen: int = field(
        default=1024,
        metadata={
            'help': 'The maximum length of the model inputs used for quant.'
        },
    )

    def __post_init__(self):
        if self.quant_bit is not None:
            assert self.quant_bit in [
                4,
                8,
            ], 'We only accept 4-bit or 8-bit quant.'
        if self.quant_type is not None:
            assert self.quant_type in [
                'nf4',
                'fp4',
            ], 'We only accept `nf4` or `fp4` quant type.'
        assert self.export_quant_bit in [
            None,
            8,
            4,
            3,
            2,
        ], 'We only accept 2/3/4/8-bit quantization.'

        if self.export_quant_bit is not None and self.export_quant_dataset is None:
            raise ValueError(
                'Quantization dataset is necessary for exporting.')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
