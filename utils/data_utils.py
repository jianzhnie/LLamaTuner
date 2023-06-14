from typing import Any, Dict, Optional, Union

import datasets

ALPACA_PROMPT_DICT = {
    'prompt_input':
    ('Below is an instruction that describes a task, paired with an input that provides further context. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: '
     ),
    'prompt_no_input':
    ('Below is an instruction that describes a task. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n### Response: '),
}


def load_data(dataset_name: str) -> Union[Dict[str, datasets.Dataset], None]:
    """
    Load a dataset based on its name.

    Args:
        dataset_name: A string representing the name of the dataset to be loaded.

    Returns:
        A dictionary containing the loaded dataset if the dataset exists.
        None if the dataset does not exist.

    Raises:
        NotImplementedError: If the dataset name provided is not implemented yet or if
            the dataset is not released.

    Examples:
        >>> load_data('alpaca')
        {'train': Dataset(...), 'validation': Dataset(...), 'test': Dataset(...)}

    """
    if dataset_name == 'alpaca':
        return datasets.load_dataset('tatsu-lab/alpaca')
    elif dataset_name == 'alpaca-clean':
        return datasets.load_dataset('yahma/alpaca-cleaned')
    elif dataset_name == 'chip2':
        return datasets.load_dataset('laion/OIG',
                                     data_files='unified_chip2.jsonl')
    elif dataset_name == 'self-instruct':
        return datasets.load_dataset('yizhongw/self_instruct',
                                     name='self_instruct')
    elif dataset_name == 'hh-rlhf':
        return datasets.load_dataset('Anthropic/hh-rlhf')
    elif dataset_name == 'longform':
        return datasets.load_dataset('akoksal/LongForm')
    elif dataset_name == 'oasst1':
        return datasets.load_dataset('timdettmers/openassistant-guanaco')
    elif dataset_name == 'vicuna':
        raise NotImplementedError('Vicuna data was not released.')
    else:
        raise NotImplementedError(
            f'Dataset {dataset_name} not implemented yet.')


def extract_alpaca_dataset(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts input from an example in the Alpaca dataset.

    Args:
        example: A dictionary containing a single example from the Alpaca dataset.

    Returns:
        A dictionary containing the extracted input string from the example.

    Examples:
        >>> example = {'input': 'example input', 'output': 'example output'}
        >>> extract_alpaca_dataset(example)
        {'input': 'example input'}

    """
    if example.get('input', '') != '':
        prompt_format = ALPACA_PROMPT_DICT['prompt_input']
    else:
        prompt_format = ALPACA_PROMPT_DICT['prompt_no_input']
    return {'input': prompt_format.format(**example)}


def format_dataset(dataset_name: str) -> Optional[Dict[str, datasets.Dataset]]:
    """
    Formats a given dataset based on its name and format.

    Args:
        dataset_name: A string representing the name of the dataset to be formatted.

    Returns:
        A dictionary containing the formatted dataset if the dataset exists in the
        specified format.
        None if the dataset does not exist or if the format is not recognized.

    Examples:
        >>> format_dataset('alpaca')
        {'train': Dataset(...), 'validation': Dataset(...), 'test': Dataset(...)}

    """
    dataset = load_data(dataset_name)
    if dataset is None:
        return None

    if dataset_name == 'alpaca' or dataset_name == 'alpaca-clean':
        dataset = dataset.map(extract_alpaca_dataset,
                              remove_columns=['instruction'])
    elif dataset_name == 'chip2':
        dataset = dataset.map(
            lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace(
                    '<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1]
            })
    elif dataset_name == 'self-instruct':
        dataset = dataset.rename_column('prompt', 'input')
        dataset = dataset.rename_column('completion', 'output')
    elif dataset_name == 'hh-rlhf':
        dataset = dataset.map(lambda x: {'input': '', 'output': x['chosen']})
    elif dataset_name == 'oasst1':
        dataset = dataset.map(lambda x: {'input': '', 'output': x['text']})
    elif dataset_name == 'input-output':
        pass
    else:
        return None  # invalid format

    # Remove unused columns.
    dataset = dataset.remove_columns([
        col for col in dataset.column_names['train']
        if col not in ['input', 'output']
    ])

    return dataset
