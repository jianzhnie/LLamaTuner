import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class DatasetAttr(object):

    dataset_name: Optional[str] = None
    hf_hub_url: Optional[str] = None
    local_path: Optional[str] = None
    dataset_formate: Optional[str] = None
    dataset_sha1: Optional[str] = None
    load_from_local: bool = False
    multi_turn: Optional[bool] = False

    def __repr__(self) -> str:
        rep = f'dataset_name: {self.dataset_name},  hf_hub_url: {self.hf_hub_url}, local_path: {self.local_path}, data_formate:{self.dataset_formate}   load_from_local: {self.load_from_local}, multi_turn: {self.multi_turn}'
        return rep

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
    instruction_template: str = field(
        default='default',
        metadata={
            'help':
            'Which template to use for constructing prompts in training and inference.'
        })
    conversation_template: str = field(
        default='default',
        metadata={
            'help':
            'Which template to use for constructing prompts in multi-turn dataset training and inference.'
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

    def init_for_training(self):  # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset_name.split(',')]
        this_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_info_path = os.path.join(this_dir, '../..', 'data',
                                          'dataset_info.yaml')
        with open(datasets_info_path, 'r') as f:
            datasets_info = yaml.safe_load(f)

        self.datasets_list: List[DatasetAttr] = []
        for i, name in enumerate(dataset_names):
            if name not in datasets_info:
                raise ValueError('Undefined dataset {} in {}'.format(
                    name, datasets_info_path))

            dataset_attr = DatasetAttr()
            dataset_attr.dataset_name = name
            dataset_attr.dataset_formate = datasets_info[name].get(
                'data_format', None)
            dataset_attr.hf_hub_url = datasets_info[name].get(
                'hf_hub_url', None)
            dataset_attr.local_path = datasets_info[name].get(
                'local_path', None)
            dataset_attr.multi_turn = datasets_info[name].get(
                'multi_turn', False)

            if datasets_info[name]['local_path'] and os.path.exists(
                    datasets_info[name]['local_path']):
                dataset_attr.load_from_local = True

            if 'columns' in datasets_info[name]:
                dataset_attr.prompt_column = datasets_info[name][
                    'columns'].get('prompt', None)
                dataset_attr.query_column = datasets_info[name]['columns'].get(
                    'query', None)
                dataset_attr.response_column = datasets_info[name][
                    'columns'].get('response', None)
                dataset_attr.history_column = datasets_info[name][
                    'columns'].get('history', None)

            self.datasets_list.append(dataset_attr)
