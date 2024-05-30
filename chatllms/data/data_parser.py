import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import yaml

from chatllms.configs.data_args import DataArguments
from chatllms.utils.constants import DATA_CONFIG
from chatllms.utils.logger_utils import get_logger
from chatllms.utils.misc import use_modelscope

logger = get_logger(__name__)


@dataclass
class DatasetAttr:
    """Dataset attributes."""

    # basic configs
    dataset_name: Optional[str] = None
    load_from: Literal['hf_hub', 'ms_hub', 'script', 'file'] = 'hf_hub'
    # extra configs
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: bool = False
    formatting: Literal['alpaca', 'sharegpt'] = 'alpaca'
    # common configs
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None
    # alpaca columns
    prompt: Optional[str] = 'instruction'
    query: Optional[str] = 'input'
    response: Optional[str] = 'output'
    history: Optional[str] = None
    # sharegpt columns
    messages: Optional[str] = 'conversations'
    # sharegpt tags
    role_tag: Optional[str] = 'from'
    content_tag: Optional[str] = 'value'
    user_tag: Optional[str] = 'human'
    assistant_tag: Optional[str] = 'gpt'
    observation_tag: Optional[str] = 'observation'
    function_tag: Optional[str] = 'function_call'
    system_tag: Optional[str] = 'system'

    def __repr__(self) -> str:
        rep = f'{self.dataset_name}, load_from: {self.load_from}, formatting: {self.formatting}'
        return rep

    def set_attr(self,
                 key: str,
                 obj: Dict[str, Any],
                 default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))


def get_dataset_list(data_args: 'DataArguments') -> List['DatasetAttr']:
    """Get a list of dataset attributes."""
    logger.info('You have set the --dataset with %s', data_args.dataset)
    file_path = os.path.join(data_args.dataset_dir, DATA_CONFIG)
    logger.info('Loading dataset information from %s...', file_path)
    if data_args.dataset is not None:
        dataset_names = [ds.strip() for ds in data_args.dataset.split(',')]
    else:
        dataset_names = []
    if data_args.interleave_probs is not None:
        data_args.interleave_probs = [
            float(prob.strip())
            for prob in data_args.interleave_probs.split(',')
        ]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset_infos = yaml.safe_load(f)
    except FileNotFoundError as err:
        error_message = f'Cannot open {file_path} due to {str(err)}.'
        raise ValueError(error_message) from err
    dataset_list: List[DatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_infos:
            raise ValueError(f'Undefined dataset {name} in {DATA_CONFIG}.')
        dataset_info = dataset_infos[name]
        has_hf_url = 'hf_hub_url' in dataset_info
        has_ms_url = 'ms_hub_url' in dataset_info

        if has_hf_url or has_ms_url:
            if (use_modelscope() and has_ms_url) or (not has_hf_url):
                dataset_attr = DatasetAttr(
                    dataset_name=dataset_info['ms_hub_url'],
                    load_from='ms_hub')
            else:
                dataset_attr = DatasetAttr(
                    dataset_name=dataset_info['hf_hub_url'],
                    load_from='hf_hub')
        elif 'script_url' in dataset_info:
            dataset_attr = DatasetAttr(dataset_name=dataset_info['script_url'],
                                       load_from='script')
        else:
            dataset_attr = DatasetAttr(dataset_name=dataset_info['file_name'],
                                       load_from='file')
        dataset_attr.set_attr('subset', dataset_info)
        dataset_attr.set_attr('folder', dataset_info)
        dataset_attr.set_attr('ranking', dataset_info, default=False)
        dataset_attr.set_attr('formatting', dataset_info, default='alpaca')
        if 'columns' in dataset_info:
            column_names = [
                'system',
                'tools',
                'images',
                'chosen',
                'rejected',
                'kto_tag',
            ]
            if dataset_attr.formatting == 'alpaca':
                column_names.extend(['prompt', 'query', 'response', 'history'])
            else:
                column_names.extend(['messages'])

            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info['columns'])

        if dataset_attr.formatting == 'sharegpt' and 'tags' in dataset_info:
            tag_names = (
                'role_tag',
                'content_tag',
                'user_tag',
                'assistant_tag',
                'observation_tag',
                'function_tag',
                'system_tag',
            )
            for tag in tag_names:
                dataset_attr.set_attr(tag, dataset_info['tags'])

        dataset_list.append(dataset_attr)
    return dataset_list
