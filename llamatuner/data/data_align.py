import os
from functools import partial
from typing import Any, Callable, Dict, List, Union

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
        content.append(example[dataset_attr.prompt])
    if dataset_attr.query and example[dataset_attr.query]:
        content.append(example[dataset_attr.query])

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


def convert_alpaca(
    examples: Dict[str, List[Any]],
    dataset_attr: DatasetAttr,
    data_args: DataArguments,
) -> Dict[str, List[Any]]:
    """
    Converts an alpaca format dataset to the standard format.

    Args:
        examples (Dict[str, List[Any]]): The dataset examples to be converted.
        dataset_attr (DatasetAttr): Attributes of the dataset.
        data_args (DataArguments): Arguments related to data processing.
        logger (Optional[logging.Logger]): Logger for logging information and errors.

    Returns:
        Dict[str, List[Any]]: The converted dataset in the standard format.
    """
    # Initialize the outputs dictionary to store the converted data
    outputs = {
        'prompt': [],
        'response': [],
        'system': [],
        'tools': [],
        'images': []
    }

    # Prepare the image conversion function with partial application
    convert_images: Callable[[Any], Any] = partial(_convert_images,
                                                   dataset_attr=dataset_attr,
                                                   data_args=data_args)

    # Iterate over each example in the dataset
    for i in range(len(examples[dataset_attr.prompt])):
        prompt = []

        # Process the conversation history if available
        if dataset_attr.history and isinstance(
                examples[dataset_attr.history][i], list):
            for old_prompt, old_response in examples[dataset_attr.history][i]:
                prompt.append({'role': Role.USER, 'content': old_prompt})
                prompt.append({
                    'role': Role.ASSISTANT,
                    'content': old_response
                })

        content = []
        # Add prompt and query to the content list
        if dataset_attr.prompt and examples[dataset_attr.prompt][i]:
            content.append(examples[dataset_attr.prompt][i])
        if dataset_attr.query and examples[dataset_attr.query][i]:
            content.append(examples[dataset_attr.query][i])

        # Append the final prompt content
        prompt.append({'role': Role.USER, 'content': '\n'.join(content)})

        # Determine the response format based on dataset attributes
        # Example: kto_tag, ranking, response
        if dataset_attr.kto_tag and isinstance(
                examples[dataset_attr.kto_tag][i], bool):
            response = [{
                'role': Role.ASSISTANT,
                'content': examples[dataset_attr.response][i],
            }]
            if examples[dataset_attr.kto_tag][i]:
                response = response + [{'role': Role.ASSISTANT, 'content': ''}]
            else:
                response = [{'role': Role.ASSISTANT, 'content': ''}] + response
        # Example: ranking, chosen, rejected
        elif (dataset_attr.ranking
              and isinstance(examples[dataset_attr.chosen][i], str)
              and isinstance(examples[dataset_attr.rejected][i], str)):
            response = [
                {
                    'role': Role.ASSISTANT,
                    'content': examples[dataset_attr.chosen][i],
                },
                {
                    'role': Role.ASSISTANT,
                    'content': examples[dataset_attr.rejected][i],
                },
            ]
        # Normal alpaca example
        elif dataset_attr.response and isinstance(
                examples[dataset_attr.response][i], str):
            response = [{
                'role': Role.ASSISTANT,
                'content': examples[dataset_attr.response][i],
            }]
        else:
            # Unsupervised example
            response = []

        # Append system, tools, and images if available
        outputs['prompt'].append(prompt)
        outputs['response'].append(response)
        outputs['system'].append(
            examples[dataset_attr.system][i] if dataset_attr.system else '')
        outputs['tools'].append(
            examples[dataset_attr.tools][i] if dataset_attr.tools else '')
        outputs['images'].append(
            convert_images(examples[dataset_attr.images][i]) if dataset_attr.
            images else [])

    return outputs


def convert_sharegpt(
    examples: Dict[str, List[Any]],
    dataset_attr: DatasetAttr,
    data_args: DataArguments,
) -> Dict[str, List[Any]]:
    """
    Converts a sharegpt format dataset to the standard format.

    Args:
        examples (Dict[str, List[Any]]): The dataset examples to be converted.
        dataset_attr (DatasetAttr): Attributes of the dataset.
        data_args (DataArguments): Arguments related to data processing.

    Returns:
        Dict[str, List[Any]]: The converted dataset in the standard format.
    """
    outputs = {
        'prompt': [],
        'response': [],
        'system': [],
        'tools': [],
        'images': []
    }
    convert_images: Callable[[Any], Any] = partial(_convert_images,
                                                   dataset_attr=dataset_attr,
                                                   data_args=data_args)

    # Tag mapping for role conversion
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

    # Iterate over each set of messages in the examples
    for i, messages in enumerate(examples[dataset_attr.messages]):
        # Extract system message if present
        if (dataset_attr.system_tag and messages[0][dataset_attr.role_tag]
                == dataset_attr.system_tag):
            system = messages[0][dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = examples[
                dataset_attr.system][i] if dataset_attr.system else ''

        if len(messages) == 0:
            continue

        aligned_messages = []
        broken_data = False

        # Align messages based on roles
        for turn_idx, message in enumerate(messages):
            if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning('Invalid role tag in {}.'.format(messages))
                broken_data = True

            aligned_messages.append({
                'role':
                tag_mapping[message[dataset_attr.role_tag]],
                'content':
                message[dataset_attr.content_tag],
            })

        # Check for valid message count
        if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
                dataset_attr.ranking and len(aligned_messages) % 2 == 0):
            logger.warning('Invalid message count in {}.'.format(messages))
            broken_data = True

        # Process the response based on dataset attributes
        if dataset_attr.kto_tag and isinstance(
                examples[dataset_attr.kto_tag][i], bool):
            # KTO example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if examples[dataset_attr.kto_tag][i]:
                response = response + [{'role': Role.ASSISTANT, 'content': ''}]
            else:
                response = [{'role': Role.ASSISTANT, 'content': ''}] + response
        elif (dataset_attr.ranking
              and isinstance(examples[dataset_attr.chosen][i], dict)
              and isinstance(examples[dataset_attr.rejected][i], dict)):
            # Ranking example, pairwise chosen and rejected
            chosen = examples[dataset_attr.chosen][i]
            rejected = examples[dataset_attr.rejected][i]
            if (chosen[dataset_attr.role_tag] not in accept_tags[-1]
                    or rejected[dataset_attr.role_tag] not in accept_tags[-1]):
                logger.warning('Invalid role tag in {}.'.format(
                    [chosen, rejected]))
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    'role': tag_mapping[chosen[dataset_attr.role_tag]],
                    'content': chosen[dataset_attr.content_tag],
                },
                {
                    'role': tag_mapping[rejected[dataset_attr.role_tag]],
                    'content': rejected[dataset_attr.content_tag],
                },
            ]
        else:
            # Normal share gpt example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        if broken_data:
            logger.warning('Skipping this abnormal example.')
            continue

        outputs['prompt'].append(prompt)
        outputs['response'].append(response)
        outputs['system'].append(system)
        outputs['tools'].append(
            examples[dataset_attr.tools][i] if dataset_attr.tools else '')
        outputs['images'].append(
            convert_images(examples[dataset_attr.images][i]) if dataset_attr.
            images else [])
    return outputs


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
        batched=True,
        remove_columns=column_names,
        **kwargs,
    )

    return aligned_dataset
