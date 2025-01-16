import os
from functools import partial
from typing import Any, Dict, List, Union

from datasets import Dataset, IterableDataset

from llamatuner.configs import DataArguments
from llamatuner.data.data_parser import DatasetAttr
from llamatuner.data.utils import Role
from llamatuner.utils.logger_utils import get_logger

logger = get_logger('llamatuner')


def _convert_images(images: List[Any], dataset_attr: DatasetAttr,
                    data_args: DataArguments) -> List[Any]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    if not isinstance(images, list):
        images = [images]
    elif len(images) == 0:
        return None
    else:
        images = images[:]

    if dataset_attr.load_from in ['script', 'file']:
        for i in range(len(images)):
            if isinstance(images[i], str) and os.path.isfile(
                    os.path.join(data_args.image_dir, images[i])):
                images[i] = os.path.join(data_args.image_dir, images[i])

    return images


def alpaca_map_fn(example: Dict[str, List[Any]], dataset_attr: DatasetAttr,
                  data_args: DataArguments):
    prompt = []
    # Process the conversation history if available
    if dataset_attr.history and isinstance(example[dataset_attr.history],
                                           list):
        for old_prompt, old_response in example[dataset_attr.history]:
            prompt.append({'role': Role.USER, 'content': old_prompt})
            prompt.append({'role': Role.ASSISTANT, 'content': old_response})

    content = []
    # Add prompt and query to the content list
    if dataset_attr.prompt and example[dataset_attr.prompt]:
        prompt_content = example[dataset_attr.prompt]
        # Convert to string if it's a list
        if isinstance(prompt_content, list):
            prompt_content = '\n'.join(map(str, prompt_content))
        content.append(prompt_content)

    if dataset_attr.query and example[dataset_attr.query]:
        # Convert to string if it's a list
        query_content = example[dataset_attr.query]
        if isinstance(query_content, list):
            query_content = '\n'.join(map(str, query_content))
        content.append(query_content)

    # Append the final prompt content
    prompt.append({'role': Role.USER, 'content': '\n'.join(content)})

    # Determine the response format based on dataset attributes
    # Example: kto_tag, ranking, response
    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag],
                                           bool):
        response = [
            {
                'role': Role.ASSISTANT,
                'content': example[dataset_attr.response]
            },
        ]
        if example[dataset_attr.kto_tag]:
            response = response + [{'role': Role.ASSISTANT, 'content': ''}]
        else:
            response = [{'role': Role.ASSISTANT, 'content': ''}] + response
    # Example: ranking, chosen, rejected
    elif (dataset_attr.ranking
          and isinstance(example[dataset_attr.chosen], str)
          and isinstance(example[dataset_attr.rejected], str)):
        response = [
            {
                'role': Role.ASSISTANT,
                'content': example[dataset_attr.chosen]
            },
            {
                'role': Role.ASSISTANT,
                'content': example[dataset_attr.rejected]
            },
        ]
    # Normal alpaca example
    elif dataset_attr.response and isinstance(example[dataset_attr.response],
                                              str):
        response = [{
            'role': Role.ASSISTANT,
            'content': example[dataset_attr.response]
        }]
    else:
        # Unsupervised example
        response = []

    convert_images = partial(_convert_images,
                             dataset_attr=dataset_attr,
                             data_args=data_args)

    output = {
        '_prompt':
        prompt,
        '_response':
        response,
        '_system':
        example[dataset_attr.system] if dataset_attr.system else '',
        '_tools':
        example[dataset_attr.tools] if dataset_attr.tools else '',
        '_images':
        convert_images(example[dataset_attr.images])
        if dataset_attr.images else None,
    }
    return output


def sharegpt_map_fn(
    example: Dict[str, Any],
    dataset_attr: DatasetAttr,
    data_args: DataArguments,
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER,
        dataset_attr.assistant_tag: Role.ASSISTANT,
        dataset_attr.observation_tag: Role.OBSERVATION,
        dataset_attr.function_tag: Role.FUNCTION,
        dataset_attr.system_tag: Role.SYSTEM,
    }
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    messages = example[dataset_attr.messages]
    if (dataset_attr.system_tag and len(messages) != 0
            and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = example[dataset_attr.system] if dataset_attr.system else ''

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning_rank0(f'Invalid role tag in {messages}.')
            broken_data = True

        aligned_messages.append({
            'role':
            tag_mapping[message[dataset_attr.role_tag]],
            'content':
            message[dataset_attr.content_tag]
        })

    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            dataset_attr.ranking and len(aligned_messages) % 2 == 0):
        logger.warning_rank0(f'Invalid message count in {messages}.')
        broken_data = True

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag],
                                           bool):  # kto example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]
        if example[dataset_attr.kto_tag]:
            response = response + [{'role': Role.ASSISTANT, 'content': ''}]
        else:
            response = [{'role': Role.ASSISTANT, 'content': ''}] + response
    elif (dataset_attr.ranking
          and isinstance(example[dataset_attr.chosen], dict) and isinstance(
              example[dataset_attr.rejected], dict)):  # pairwise example
        chosen = example[dataset_attr.chosen]
        rejected = example[dataset_attr.rejected]
        if (chosen[dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[dataset_attr.role_tag] not in accept_tags[-1]):
            logger.warning_rank0(f'Invalid role tag in {[chosen, rejected]}.')
            broken_data = True

        prompt = aligned_messages
        response = [
            {
                'role': tag_mapping[chosen[dataset_attr.role_tag]],
                'content': chosen[dataset_attr.content_tag]
            },
            {
                'role': tag_mapping[rejected[dataset_attr.role_tag]],
                'content': rejected[dataset_attr.content_tag]
            },
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning_rank0('Skipping this abnormal example.')
        prompt, response = [], []

    convert_images = partial(_convert_images,
                             dataset_attr=dataset_attr,
                             data_args=data_args)
    output = {
        '_prompt':
        prompt,
        '_response':
        response,
        '_system':
        system,
        '_tools':
        example[dataset_attr.tools] if dataset_attr.tools else '',
        '_images':
        convert_images(example[dataset_attr.images])
        if dataset_attr.images else None
    }
    return output


def align_dataset(
    dataset: Union[Dataset, IterableDataset],
    dataset_attr: DatasetAttr,
    data_args: DataArguments,
) -> Union[Dataset, IterableDataset]:
    """
    Aligns the dataset to the specified format.

    Args:
        dataset (Union[Dataset, IterableDataset]): The input dataset to be aligned.
        dataset_attr (DatasetAttr): Attributes of the dataset specifying its format and columns.
        data_args (DataArguments): Arguments related to data processing.

    Returns:
        Union[Dataset, IterableDataset]: The aligned dataset.
    """
    logger.info(dataset_attr)
    # Determine the conversion function based on the dataset formatting
    if dataset_attr.formatting == 'alpaca':
        convert_func = partial(alpaca_map_fn,
                               dataset_attr=dataset_attr,
                               data_args=data_args)
    else:
        convert_func = partial(sharegpt_map_fn,
                               dataset_attr=dataset_attr,
                               data_args=data_args)

    # Get the column names from the dataset
    column_names = list(next(iter(dataset)).keys())

    # Set additional arguments for the dataset map function
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc='Converting format of dataset',
        )

    # Apply the conversion function to the dataset
    aligned_dataset = dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )

    return aligned_dataset
