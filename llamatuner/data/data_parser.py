import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

import yaml

from llamatuner.configs.data_args import DataArguments
from llamatuner.utils.constants import DATA_CONFIG
from llamatuner.utils.logger_utils import get_logger
from llamatuner.utils.misc import use_modelscope

logger = get_logger('llamatuner')


def get_attrs(cls):
    attrs = vars(cls)
    return {k: v for k, v in attrs.items() if not k.startswith('__')}


@dataclass
class DatasetAttr:
    """
    Dataset attributes configuration.

    Attributes:
        dataset_name (Optional[str]): Name or URL of the dataset.
        load_from (Literal['hf_hub', 'ms_hub', 'script', 'file']): Source to load the dataset from.
        ranking (bool): Whether the dataset involves ranking.
        subset (Optional[str]): Subset of the dataset.
        folder (Optional[str]): Folder containing the dataset.
        formatting (Literal['alpaca', 'sharegpt']): Formatting style of the dataset.
        system (Optional[str]): System-related information column.
        tools (Optional[str]): Tools-related information column.
        images (Optional[str]): Images-related information column.
        chosen (Optional[str]): Column for chosen responses.
        rejected (Optional[str]): Column for rejected responses.
        kto_tag (Optional[str]): Column for KTO tags.
        prompt (Optional[str]): Column for prompts (Alpaca formatting).
        query (Optional[str]): Column for queries (Alpaca formatting).
        response (Optional[str]): Column for responses (Alpaca formatting).
        history (Optional[str]): Column for history (Alpaca formatting).
        messages (Optional[str]): Column for messages (ShareGPT formatting).
        role_tag (Optional[str]): Column for role tags (ShareGPT formatting).
        content_tag (Optional[str]): Column for content tags (ShareGPT formatting).
        user_tag (Optional[str]): Column for user tags (ShareGPT formatting).
        assistant_tag (Optional[str]): Column for assistant tags (ShareGPT formatting).
        observation_tag (Optional[str]): Column for observation tags (ShareGPT formatting).
        function_tag (Optional[str]): Column for function call tags (ShareGPT formatting).
        system_tag (Optional[str]): Column for system tags (ShareGPT formatting).
    """

    # Basic configs
    dataset_name: Optional[str] = None
    load_from: Literal['hf_hub', 'ms_hub', 'script', 'file'] = 'hf_hub'
    formatting: Literal['alpaca', 'sharegpt'] = 'alpaca'

    # Extra configs
    ranking: bool = False
    subset: Optional[str] = None
    split: str = 'train'
    folder: Optional[str] = None
    num_samples: Optional[int] = None

    # Common configs
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None

    # RLHF columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None

    # Alpaca columns
    prompt: Optional[str] = 'instruction'
    query: Optional[str] = 'input'
    response: Optional[str] = 'output'
    history: Optional[str] = None

    # ShareGPT columns
    messages: Optional[str] = 'conversations'

    # ShareGPT tags
    role_tag: Optional[str] = 'from'
    content_tag: Optional[str] = 'value'
    user_tag: Optional[str] = 'human'
    assistant_tag: Optional[str] = 'gpt'
    observation_tag: Optional[str] = 'observation'
    function_tag: Optional[str] = 'function_call'
    system_tag: Optional[str] = 'system'

    def __repr__(self) -> str:
        return f'DatasetAttr(name: {self.dataset_name}, load_from: {self.load_from}, formatting: {self.formatting}, split: {self.split})'

    def set_attr(self,
                 key: str,
                 obj: Dict[str, Any],
                 default: Optional[Any] = None) -> None:
        """Set an attribute from a dictionary with an optional default value."""
        setattr(self, key, obj.get(key, default))


def get_dataset_attr_list(dataset_names: Optional[Sequence[str]],
                          data_args: DataArguments) -> List[DatasetAttr]:
    """
    Get a list of dataset attributes based on the provided dataset arguments.

    Args:
        dataset_names: Sequence of dataset names to process.
        data_args (DataArguments): The dataset arguments containing dataset information.

    Returns:
        List[DatasetAttr]: A list of DatasetAttr objects with configured attributes.
    """

    file_path = os.path.join(data_args.dataset_dir, DATA_CONFIG)
    if dataset_names is None:
        dataset_names = []

    if not dataset_names:
        raise ValueError(
            f'No dataset specified in the --dataset argument, please refer to the {file_path} '
            'file for available datasets.')

    # Parse interleave probabilities if provided
    if data_args.interleave_probs:
        data_args.interleave_probs = [
            float(prob.strip())
            for prob in data_args.interleave_probs.split(',')
        ]

    # Load dataset configuration
    try:
        logger.info('Loading dataset information config file from %s...',
                    file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset_infos = yaml.safe_load(file)
    except FileNotFoundError as err:
        error_message = f'Cannot open {file_path} due to {str(err)}.'
        raise ValueError(error_message) from err

    def _create_dataset_attr(name: str, info: Dict[str, Any]) -> DatasetAttr:
        """Helper function to create and configure a DatasetAttr instance.

        Args:
            name: Name of the dataset.
            info: Dataset configuration information.

        Returns:
            Configured DatasetAttr instance.
        """
        # Determine data source and create base DatasetAttr
        has_hf_url = 'hf_hub_url' in info
        has_ms_url = 'ms_hub_url' in info

        if has_hf_url or has_ms_url:
            if (use_modelscope() and has_ms_url) or (not has_hf_url):
                dataset_attr = DatasetAttr(dataset_name=info['ms_hub_url'],
                                           load_from='ms_hub')
            else:
                dataset_attr = DatasetAttr(dataset_name=info['hf_hub_url'],
                                           load_from='hf_hub')
        elif 'script_url' in info:
            dataset_attr = DatasetAttr(dataset_name=info['script_url'],
                                       load_from='script')
        else:
            dataset_attr = DatasetAttr(dataset_name=info['file_name'],
                                       load_from='file')

        # Configure basic attributes
        for attr in ['subset', 'folder', 'num_samples']:
            dataset_attr.set_attr(attr, info)

        dataset_attr.set_attr('ranking', info, default=False)
        dataset_attr.set_attr('formatting', info, default='alpaca')
        dataset_attr.set_attr('split', info, default='train')

        # Configure column names
        if 'columns' in info:
            _configure_columns(dataset_attr, info['columns'])

        # Configure ShareGPT tags
        if dataset_attr.formatting == 'sharegpt' and 'tags' in info:
            _configure_tags(dataset_attr, info['tags'])

        return dataset_attr

    def _configure_columns(dataset_attr: DatasetAttr,
                           columns: Dict[str, str]) -> None:
        """Configure dataset columns based on formatting type."""
        base_columns = [
            'system', 'tools', 'images', 'videos', 'chosen', 'rejected',
            'kto_tag'
        ]
        format_columns = ([
            'prompt', 'query', 'response', 'history'
        ] if dataset_attr.formatting == 'alpaca' else ['messages'])

        for column in base_columns + format_columns:
            dataset_attr.set_attr(column, columns)

    def _configure_tags(dataset_attr: DatasetAttr, tags: Dict[str,
                                                              str]) -> None:
        """Configure ShareGPT specific tags."""
        tag_names = [
            'role_tag', 'content_tag', 'user_tag', 'assistant_tag',
            'observation_tag', 'function_tag', 'system_tag'
        ]
        for tag in tag_names:
            dataset_attr.set_attr(tag, tags)

    # Process each dataset
    dataset_attr_list: List[DatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_infos:
            raise ValueError(
                f'Undefined dataset {name} in dataset config file {file_path}.'
            )
        dataset_attr_ = _create_dataset_attr(name, dataset_infos[name])
        dataset_attr_list.append(dataset_attr_)

    return dataset_attr_list
