from dataclasses import dataclass, field
from typing import Optional


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
        default=None,
        metadata={
            'help': 'Which dataset to finetune on. See datamodule for options.'
        })
    data_path: str = field(
        default=None,
        metadata={
            'help':
            'where is dataset in local dir. See datamodule for options.'
        })
    # 数据集的本地路径，如果load_from_local为True，那么就从本地加载数据集
    data_dir: str = field(
        default=None,
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
    prompt_template: str = field(
        default='instruction',
        metadata={
            'help':
            'Which template to use for constructing prompts in training and inference.'
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

    def __post_init__(self):
        if self.dataset_name is None and self.data_path and self.data_dir is None:
            raise ValueError('Need either a dataset name or a data_path .')
        else:
            if self.data_path is not None:
                extension = self.data_path.split('.')[-1]
                assert extension in [
                    'json', 'jsonl'
                ], '`train_file` should be a json or a jsonl file.'
