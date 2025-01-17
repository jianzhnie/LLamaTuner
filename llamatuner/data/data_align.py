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
    r"""Convert image paths to full paths when loading from local disk.

    Args:
        images: Single image or list of images
        dataset_attr: Dataset attributes containing load source
        data_args: Data arguments containing image directory

    Returns:
        List of processed images or None if empty
    """
    if not isinstance(images, list):
        images = [images]
    if not images:
        return None

    images = images[:]
    if dataset_attr.load_from in ['script', 'file']:
        for i in range(len(images)):
            img_name = images[i]
            if isinstance(img_name, str) and os.path.isfile(
                    os.path.join(data_args.image_dir, img_name)):
                images[i] = os.path.join(data_args.image_dir, img_name)

    return images


def alpaca_map_fn(example: Dict[str, List[Any]], dataset_attr: DatasetAttr,
                  data_args: DataArguments):
    """Convert dataset to standardized Alpaca format.

    Args:
        example: Single example from dataset containing conversation data
        dataset_attr: Dataset attributes specifying format and column names
        data_args: Data processing arguments

    Returns:
        Dict containing standardized conversation format with prompts, responses,
        system message, tools, and images
    """
    prompt: List[Dict[str, str]] = []
    # Process the conversation history if available
    if dataset_attr.history and isinstance(example[dataset_attr.history],
                                           list):
        for old_prompt, old_response in example[dataset_attr.history]:
            prompt.append({'role': Role.USER, 'content': str(old_prompt)})
            prompt.append({
                'role': Role.ASSISTANT,
                'content': str(old_response)
            })

    # Combine prompt and query content
    content: List[str] = []
    for field in [dataset_attr.prompt, dataset_attr.query]:
        if field and example.get(field):
            field_content = example[field]
            if isinstance(field_content, list):
                field_content = '\n'.join(map(str, field_content))
            content.append(str(field_content))

    # Add final user prompt
    if content:
        prompt.append({'role': Role.USER, 'content': '\n'.join(content)})

    # Determine the response format based on dataset attributes
    # Example: kto_tag, ranking, response
    response: List[Dict[str, str]] = []
    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag],
                                           bool):
        # Knowledge-testing output format
        response = [
            {
                'role': Role.ASSISTANT,
                'content': str(example[dataset_attr.response])
            },
        ]
        empty_response = {'role': Role.ASSISTANT, 'content': ''}

        response = [empty_response] + response if not example[
            dataset_attr.kto_tag] else response + [empty_response]

    # Example: ranking, chosen, rejected
    elif (dataset_attr.ranking
          and isinstance(example[dataset_attr.chosen], str)
          and isinstance(example[dataset_attr.rejected], str)):

        # Ranking format with chosen/rejected responses
        response = [
            {
                'role': Role.ASSISTANT,
                'content': str(example[dataset_attr.chosen])
            },
            {
                'role': Role.ASSISTANT,
                'content': str(example[dataset_attr.rejected])
            },
        ]
    # Normal alpaca example
    elif dataset_attr.response and isinstance(example[dataset_attr.response],
                                              str):
        # Standard single response format
        response = [{
            'role': Role.ASSISTANT,
            'content': str(example[dataset_attr.response])
        }]

    convert_images = partial(_convert_images,
                             dataset_attr=dataset_attr,
                             data_args=data_args)

    output = {
        '_prompt':
        prompt,
        '_response':
        response,
        '_system':
        str(example[dataset_attr.system]) if dataset_attr.system else '',
        '_tools':
        str(example[dataset_attr.tools]) if dataset_attr.tools else '',
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
    r"""Convert  dataset to standardized ShareGPT format.

    Args:
        example: Single example containing conversation messages
        dataset_attr: Dataset attributes specifying format and column names
        data_args: Data processing arguments

    Returns:
        Dict containing standardized conversation format with prompts, responses,
        system message, tools, and images

    Note:
        ShareGPT format expects alternating user/assistant messages with optional
        system message at start. Messages are validated for correct role ordering.
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

    # Extract messages and handle system message
    messages = example[dataset_attr.messages]
    system = ''
    if (dataset_attr.system_tag and messages
            and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag):
        system = str(messages[0][dataset_attr.content_tag])
        messages = messages[1:]
    elif dataset_attr.system:
        system = example[dataset_attr.system]

    # Process and validate messages
    aligned_messages: List[Dict[str, str]] = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning(f'Invalid role tag in {messages}.')
            broken_data = True

        aligned_messages.append({
            'role':
            tag_mapping[message[dataset_attr.role_tag]],
            'content':
            message[dataset_attr.content_tag]
        })

    # Validate message count
    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            dataset_attr.ranking and len(aligned_messages) % 2 == 0):
        logger.warning(f'Invalid message count in {messages}.')
        broken_data = True

    # Handle different response formats
    prompt: List[Dict[str, str]] = []
    response: List[Dict[str, str]] = []

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag],
                                           bool):  # kto example
        # Knowledge-testing format
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]
        empty_response = {'role': Role.ASSISTANT, 'content': ''}

        response = [empty_response] + response if not example[
            dataset_attr.kto_tag] else response + [empty_response]

    elif (dataset_attr.ranking
          and isinstance(example[dataset_attr.chosen], dict) and isinstance(
              example[dataset_attr.rejected], dict)):  # pairwise example
        # Ranking format
        chosen = example[dataset_attr.chosen]
        rejected = example[dataset_attr.rejected]
        if (chosen[dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[dataset_attr.role_tag] not in accept_tags[-1]):
            logger.warning(f'Invalid role tag in {[chosen, rejected]}.')
            broken_data = True

        prompt = aligned_messages
        response = [
            {
                'role': tag_mapping[chosen[dataset_attr.role_tag]],
                'content': str(chosen[dataset_attr.content_tag])
            },
            {
                'role': tag_mapping[rejected[dataset_attr.role_tag]],
                'content': str(rejected[dataset_attr.content_tag])
            },
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning('Skipping this abnormal example.')

    # Process images if present
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
        str(example[dataset_attr.tools]) if dataset_attr.tools else '',
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
    logger.info(f'Aligning dataset with attributes: {dataset_attr}')
    # Determine the conversion function based on the dataset formatting
    if dataset_attr.formatting not in ['alpaca', 'sharegpt']:
        raise ValueError(
            f'Unsupported dataset format: {dataset_attr.formatting}')
    if dataset_attr.formatting == 'alpaca':
        convert_func = partial(alpaca_map_fn,
                               dataset_attr=dataset_attr,
                               data_args=data_args)
    else:
        convert_func = partial(sharegpt_map_fn,
                               dataset_attr=dataset_attr,
                               data_args=data_args)

    # Get column names safely
    try:
        column_names = list(next(iter(dataset)).keys())
    except StopIteration:
        logger.warning('Empty dataset provided')
        return dataset

    # Set additional arguments for the dataset map function
    kwargs = {}
    if not data_args.streaming:
        kwargs = {
            'num_proc': data_args.preprocessing_num_workers,
            'load_from_cache_file': not data_args.overwrite_cache,
            'desc': 'Converting format of dataset',
        }

    # Apply the conversion function to the dataset
    aligned_dataset = dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )

    return aligned_dataset
