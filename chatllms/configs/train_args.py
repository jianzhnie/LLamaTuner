from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class TrainingArguments(TrainingArguments):
    # 缓存目录
    cache_dir: Optional[str] = field(default=None)
    # 不使用adapter进行全微调（不适用Lora或qlora？）
    full_finetune: bool = field(
        default=False,
        metadata={'help': 'Finetune the entire model without adapters.'})
    # 是否进行训练，那肯定是要的
    do_train: bool = field(
        default=True,
        metadata={'help': 'To train or not to train, that is the question?'})
    # 是否进行验证
    do_eval: bool = field(
        default=False,
        metadata={'help': 'To train or not to train, that is the question?'})
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
