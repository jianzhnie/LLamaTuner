import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class DatasetAttr(object):

    load_from: str
    dataset_name: Optional[str] = None
    dataset_sha1: Optional[str] = None
    dataset_path: Optional[str] = None

    def __repr__(self) -> str:
        return self.dataset_name

    def __post_init__(self):
        self.prompt_column = 'instruction'
        self.query_column = 'input'
        self.response_column = 'output'
        self.history_column = None


@dataclass
class DataArguments:
    # 微调数据集是 alpaca
    dataset_name: Optional[str] = field(
        default='alpaca',
        metadata={
            'help': 'Which dataset to finetune on. See datamodule for options.'
        })
    # 数据集的本地路径，如果load_from_local为True，那么就从本地加载数据集
    dataset_dir: str = field(
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
    multiturn_dialogue: Optional[str] = field(
        default=False,
        metadata={
            'help': 'The dataset is a multiturn_dialogue dataset or not'
        })
    lazy_preprocess: bool = True
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

    def init_for_training(self):  # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset_name.split(',')]
        this_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_info_path = os.path.join(this_dir, '../..', 'data',
                                         'dataset_info.yaml')
        with open(dataset_info_path, 'r') as f:
            dataset_info = yaml.safe_load(f)

        self.dataset_list: List[DatasetAttr] = []
        for i, name in enumerate(dataset_names):
            if name not in dataset_info:
                raise ValueError('Undefined dataset {} in {}'.format(
                    name, dataset_info_path))

            if 'hf_hub_url' in dataset_info[name]:
                dataset_attr = DatasetAttr(
                    'hf_hub', dataset_name=dataset_info[name]['hf_hub_url'])
            elif 'script_url' in dataset_info[name]:
                dataset_attr = DatasetAttr(
                    'script', dataset_name=dataset_info[name]['script_url'])
            elif 'local_path' in dataset_info[name]:
                dataset_attr = DatasetAttr(
                    'dataset_path',
                    dataset_name=dataset_info[name]['local_path'])

            if 'columns' in dataset_info[name]:
                dataset_attr.prompt_column = dataset_info[name]['columns'].get(
                    'prompt', None)
                dataset_attr.query_column = dataset_info[name]['columns'].get(
                    'query', None)
                dataset_attr.response_column = dataset_info[name][
                    'columns'].get('response', None)
                dataset_attr.history_column = dataset_info[name][
                    'columns'].get('history', None)

            self.dataset_list.append(dataset_attr)
