import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class DatasetAttr(object):

    dataset_name: Optional[str] = None
    hf_hub_url: Optional[str] = None
    local_path: Optional[str] = None
    dataset_format: Optional[str] = None
    load_from_local: bool = False
    multi_turn: Optional[bool] = False

    def __repr__(self) -> str:
        rep = (f'dataset_name: {self.dataset_name} || '
               f'hf_hub_url: {self.hf_hub_url} || '
               f'local_path: {self.local_path} \n'
               f'data_formate: {self.dataset_format}  || '
               f'load_from_local: {self.load_from_local} || '
               f'multi_turn: {self.multi_turn}')
        return rep

    def __post_init__(self):
        self.prompt_column = 'instruction'
        self.query_column = 'input'
        self.response_column = 'output'
        self.history_column = None


@dataclass
class DataArguments:
    dataset_cfg: Optional[str] = field(
        default='./data/alpaca_zh.yaml',
        metadata={
            'help':
            'Path to dataset infos, please refer to `./data/README.md` to see how to prepare your datasets for training.'
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
        assert self.dataset_cfg is not None and os.path.exists(
            self.dataset_cfg
        ), f'{self.dataset_cfg} does not exist!, please check the path.'
        datasets_info = yaml.safe_load(open(self.dataset_cfg, 'r'))
        self.dataset_names = list(datasets_info.keys())
        self.dataset_attr_list: List[DatasetAttr] = []
        for i, name in enumerate(self.dataset_names):
            dataset_attr = DatasetAttr()
            dataset_attr.dataset_name = name
            dataset_attr.dataset_format = datasets_info[name].get(
                'dataset_format', None)
            dataset_attr.hf_hub_url = datasets_info[name].get(
                'hf_hub_url', None)
            dataset_attr.local_path = datasets_info[name].get(
                'local_path', None)
            dataset_attr.multi_turn = datasets_info[name].get(
                'multi_turn', False)

            if datasets_info[name]['local_path'] and os.path.exists(
                    datasets_info[name]['local_path']):
                dataset_attr.load_from_local = True
            else:
                dataset_attr.load_from_local = False
                raise Warning(
                    'You have set local_path: {} for {} but it does not exist! Will load the data from {}'
                    .format(name, dataset_attr.local_path,
                            dataset_attr.hf_hub_url))

            if 'columns' in datasets_info[name]:
                dataset_attr.prompt_column = datasets_info[name][
                    'columns'].get('prompt', None)
                dataset_attr.query_column = datasets_info[name]['columns'].get(
                    'query', None)
                dataset_attr.response_column = datasets_info[name][
                    'columns'].get('response', None)
                dataset_attr.history_column = datasets_info[name][
                    'columns'].get('history', None)

            self.dataset_attr_list.append(dataset_attr)
