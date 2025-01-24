import argparse
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer

from llamatuner.data.dataset_factory.sft_dataset import (
    DataCollatorForSupervisedDataset, SupervisedDataset)
from llamatuner.data.dataset_factory.sharegpt_dataset import (
    ConversationDataset, VicunaDataset)
from llamatuner.data.template import (ALPACA_PROMPT_DICT, DEFAULT_PROMPT_DICT,
                                      RANDOM_PROMPT_DICT)


def extract_default_prompt_dataset(example: Dict[str, Any]) -> Dict[str, str]:
    # Not random, use pre-defined templates
    if example.get('input', '') != '':
        prompt_template = DEFAULT_PROMPT_DICT['prompt_input']
    else:
        prompt_template = DEFAULT_PROMPT_DICT['prompt_no_input']

    # Format prompt with example
    formated_prompt = prompt_template.format(**example)

    return {'input': formated_prompt}


def extract_alpaca_prompt_dataset(example: Dict[str, Any]) -> Dict[str, str]:
    """Extracts input from an example in the Alpaca dataset.

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


def extract_vicuna_prompt_dataset(example: Dict[str, Any]) -> Dict[str, str]:
    """Extracts the input and output portions of a single conversation example
    from the Vicuña format.

    Args:
        example (Dict[str, Any]): A single conversation example in the Vicuña format.

    Returns:
        Dict[str, str]: A dictionary containing the input and output portions of the conversation.
    """
    # Set default system message
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful,\
          detailed, and polite answers to the user's questions."

    # Define roles and role mappings
    roles = ('USER', 'ASSISTANT')
    roles_mapping = {'human': roles[0], 'gpt': roles[1]}

    # Define separators for input and output messages
    seps = [' ', '</s>']

    # Extract messages from conversation
    messages = []
    conversations = example['conversations']
    if conversations[0]['from'].lower() == 'system':
        # If first message is from system, use it as system message
        system = conversations[0]['value']
        conversations = conversations[1:]
    if roles_mapping[conversations[0]['from']] != roles[0]:
        # If first message is not from human, skip it
        conversations = conversations[1:]
    for j, sentence in enumerate(conversations):
        # Assign role based on sender
        role = roles_mapping[sentence['from']]
        assert role == roles[j % 2], f'Unexpected role at index {j}'
        messages.append((role, sentence['value']))

    # Concatenate messages into input and output portions
    ret = system + seps[0]
    for i, (role, message) in enumerate(messages):
        if message:
            ret += role + ': ' + message + seps[i % 2]
        else:
            ret += role + ':'
    sep = seps[0] + roles[1] + ': '
    input_str, output_str = ret.rsplit(sep, 1)
    input_str += sep

    return {'input': input_str, 'output': output_str}


def extract_random_prompt_dataset(example: Dict[str, Any]) -> Dict[str, str]:

    random_prompt_input = RANDOM_PROMPT_DICT['prompt_input']
    random_prompt_without_input = RANDOM_PROMPT_DICT['prompt_no_input']
    # Randomly choose prompt template
    if example.get('input', '') != '':
        # Input exists, choose from prompts with input
        prompt_template, _ = random.choices(
            random_prompt_input,
            weights=[w for _, w in random_prompt_input])[0]
    else:
        # No input, choose from prompts without input
        prompt_template, _ = random.choices(
            random_prompt_without_input,
            weights=[w for _, w in random_prompt_without_input])[0]

    # Format prompt with example
    formated_prompt = prompt_template.format(**example)

    return {'input': formated_prompt}


def load_local_dataset(
        dataset_path: str,
        eval_dataset_size: float = 0.1) -> Tuple[Dataset, Dataset]:
    """Reads in a dataset from a file and returns it as a split train-test
    dataset.

    Args:
        dataset_path (str): The name of the dataset file to read in. \
            The format is inferred based on the file extension.
        eval_dataset_size (float, optional): The fraction of the dataset to use for evaluation. Defaults to 0.1.

    Returns:
        A tuple containing two datasets - the training subset and the testing subset.
    Raises:
        ValueError: If the specified file format is unsupported.
    """

    # Read in the full dataset from file based on the file format
    if dataset_path.endswith('.json'):
        full_dataset = load_dataset('json', data_files=dataset_path)
    elif dataset_path.endswith('.jsonl'):
        full_dataset = load_dataset('json', data_files=dataset_path)
    elif dataset_path.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_path))
    elif dataset_path.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(
            pd.read_csv(dataset_path, delimiter='\t'))
    else:
        raise ValueError(f'Unsupported dataset format: {dataset_path}')
    if 'train' not in full_dataset:
        split_dataset = full_dataset.train_test_split(
            test_size=eval_dataset_size)
        return split_dataset
    else:
        return full_dataset


def load_data(
    dataset_path: str,
    eval_dataset_size: float = 0.1,
    text_logger: logging.Logger = None,
) -> Union[Dict[str, Dataset], None]:
    """Load a dataset based on its name.

    Args:
        dataset_path: A string representing the path to the dataset to be loaded.
        eval_dataset_size: A float representing the size of the evaluation dataset.
        text_logger: A logger object to log messages.

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
    if not os.path.exists(dataset_path):
        # Download dataset from HuggingFace Datasets
        text_logger.info(
            f'Lodding dataset from huggingface, please ref to https://huggingface.co/datasets/{dataset_path}'
        )
        dataset = load_dataset(dataset_path,
                               cache_dir='~/.cache/huggingface/datasets')
        return dataset
    else:
        # Load dataset from local file
        try:
            text_logger.info(
                f'Lodding dataset from local path: {dataset_path}')
            dataset = load_local_dataset(dataset_path, eval_dataset_size)
            return dataset
        except:
            raise ValueError(f'Error loading dataset from {dataset_path}')


def formate_instruction_dataset(
    dataset: Dataset,
    dataset_name: str,
    dataset_format: str,
    instruction_template: str = 'default',
    text_logger: logging.Logger = None,
) -> Optional[Dict[str, Dataset]]:
    """Formats a given dataset based on its name and format.

    Removes unused columns, renames columns to 'input' and 'output',
    and applies dataset-specific formatting based on the dataset_name.

    Returns formatted dataset dict if dataset can be formatted, else None.

    Args:
        dataset: A dataset object to be formatted.
        dataset_name: A string representing the name of the dataset to be formatted.
        dataset_format: A string representing the name of the dataset format to be used.
        instruction_template: A string representing the name of the prompt template to be used.
        text_logger: A logger object to log messages.

    Returns:
        A dictionary containing the formatted dataset if the dataset exists in the
        specified format.
        None if the dataset does not exist or if the format is not recognized.
    """

    def _format_dolly15k(dataset: Dataset) -> Dataset:
        """Format Dolly-15k dataset."""
        dataset = dataset.rename_column('context', 'input')
        dataset = dataset.rename_column('response', 'output')
        return dataset

    def _format_chip2(dataset: Dataset) -> Dataset:
        """Format CHIP-2 dataset."""
        dataset = dataset.map(
            lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace(
                    '<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1]
            })
        return dataset

    def _format_self_instruct(dataset: Dataset) -> Dataset:
        """Format Self-Instruct dataset.

        hf_url: https://huggingface.co/datasets/yizhongw/self_instruct/viewer/self_instruct/train
        """
        dataset = dataset.rename_column('prompt', 'input')
        dataset = dataset.rename_column('completion', 'output')
        return dataset

    def _format_hh_rlhf(dataset: Dataset) -> Dataset:
        """Format HH-RLHF dataset."""
        dataset = dataset.map(lambda x: {'input': '', 'output': x['chosen']})
        return dataset

    def _format_oasst1(dataset: Dataset) -> Dataset:
        """Format OASST1 dataset."""
        dataset = dataset.map(lambda x: {'input': '', 'output': x['text']})
        return dataset

    def _format_100Poison(dataset: Dataset) -> Dataset:
        """Format ShareGPT dataset."""
        dataset = dataset.rename_column('prompt', 'instruction')
        dataset = dataset.rename_column('answer', 'output')
        return dataset

    def _remove_unused_columns(dataset):
        """Remove columns not named 'input' or 'output'."""
        dataset = dataset.remove_columns([
            col for col in dataset.column_names['train']
            if col not in ['input', 'output']
        ])
        return dataset

    # Format dataset
    text_logger.info(
        f'Original {dataset_name} using {dataset_format} dataset format.')
    text_logger.info(
        f'Formatting the dataset {dataset_name} to alpaca dataset format.')
    if dataset_format == 'alpaca':
        text_logger.info('By default, We support the Alpaca dataset format.')
    elif dataset_format == 'dolly':
        dataset = _format_dolly15k(dataset)
    elif dataset_format == 'chip2':
        dataset = _format_chip2(dataset)
    elif dataset_format == 'self-instruct':
        dataset = _format_self_instruct(dataset)
    elif dataset_format == 'hh-rlhf':
        dataset = _format_hh_rlhf(dataset)
    elif dataset_format == 'oasst1':
        dataset = _format_oasst1(dataset)
    elif dataset_format == '100PoisonMpts':
        dataset = _format_100Poison(dataset)
    else:
        raise NotImplementedError(
            f'Unsupported dataset format: {dataset_format},  Please add the formate function in data_utils.py'
        )
    # encode_instruction_example
    text_logger.info(f'Applying instruction template: {instruction_template}')
    if instruction_template == 'alpaca':
        dataset = dataset.map(extract_alpaca_prompt_dataset)
    elif instruction_template == 'random':
        dataset = dataset.map(extract_random_prompt_dataset)
    else:
        dataset = dataset.map(extract_default_prompt_dataset)

    # Remove unused columns.
    text_logger.info(
        "Removing the unused columns, keep only 'input' and 'output'")
    dataset = _remove_unused_columns(dataset)

    return dataset


def split_train_eval(
    dataset: Dataset,
    do_eval: bool = False,
    eval_dataset_size: float = 0.1,
    max_eval_samples: int = None,
    do_train: bool = True,
    max_train_samples: int = None,
    text_logger: logging.Logger = None,
) -> Dict[str, Dataset]:
    """Prepare the training and evaluation datasets for a machine learning
    model.

    Args:
        dataset (DatasetDict): The complete dataset containing train, validation, and test splits.
        do_eval (bool, optional): Whether to use an evaluation dataset or not. Defaults to False.
        eval_dataset_size (float, optional): The size of the validation set if splitting from the training data.
            Ignored if `do_eval` is False. Defaults to 0.2.
        max_eval_samples (int, optional): The maximum number of samples to keep in the evaluation dataset.
            Ignored if `do_eval` is False or `None`. Defaults to None.
        do_train (bool, optional): Whether to use a training dataset or not. Defaults to True.
        max_train_samples (int, optional): The maximum number of samples to keep in the training dataset.
            Ignored if `do_train` is False or `None`. Defaults to None.
        text_logger (logging.Logger, optional): A logger object to log messages. Defaults to None.

    Returns:
        Dict[str, Dataset]: A dictionary containing the prepared training and evaluation datasets
        (if used), where the keys are 'train' and 'eval', respectively.
    """
    if not isinstance(dataset, DatasetDict):
        raise TypeError("The 'dataset' argument must be a DatasetDict object.")

    train_dataset, eval_dataset = None, None
    # Prepare evaluation dataset
    if do_eval:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            # Split train dataset in train and validation according to `eval_dataset_size`
            text_logger.info(
                f'Splitting the dataset into train and validation according to `eval_dataset_size`:  {eval_dataset_size}'
            )
            dataset = dataset['train'].train_test_split(
                test_size=eval_dataset_size, shuffle=True, seed=42)
            eval_dataset = dataset['test']

        # Reduce evaluation dataset size (if specified)
        text_logger.info(
            f'You have set the max_eval_samples: {max_eval_samples}, will do sampling ...'
        )
        if max_eval_samples is not None and len(
                eval_dataset) > max_eval_samples:
            eval_dataset = eval_dataset.select(np.arange(max_eval_samples))

    # Prepare training dataset
    if do_train:
        train_dataset = dataset['train']

        # Reduce training dataset size (if specified)
        text_logger.info(
            f'You have set the max_train_samples: {max_train_samples}, will do sampling ...'
        )
        if max_train_samples is not None and len(
                train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(np.arange(max_train_samples))

    return train_dataset, eval_dataset


def make_data_module(args, text_logger):
    """Make dataset and collator for supervised fine-tuning. Datasets are
    expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - vicuna
    """
    train_datasets: List[Dataset] = []
    eval_datasets: List[Dataset] = []
    dataset_name_list = args.dataset_names
    text_logger.info(f'Loading datasets: {dataset_name_list}')
    mutliturn_lst = [
        dataset_attr.multi_turn for dataset_attr in args.dataset_attr_list
    ]
    assert mutliturn_lst.count(mutliturn_lst[0]) == len(
        mutliturn_lst
    ), 'All datasets should be multi-turn or single-turn. As follwing we will concat all datasets, so they should be in the same format.'

    for dataset_attr in args.dataset_attr_list:
        text_logger.info('=' * 80)
        text_logger.info('DatasetAttr: {}'.format(dataset_attr))

        if dataset_attr.load_from_local:
            dataset_path = dataset_attr.local_path
        elif dataset_attr.hf_hub_url:
            dataset_path = dataset_attr.hf_hub_url
        else:
            raise ValueError('Please set the dataset path or hf_hub_url.')

        dataset = load_data(
            dataset_path,
            eval_dataset_size=args.eval_dataset_size,
            text_logger=text_logger,
        )

        if not dataset_attr.multi_turn:
            dataset = formate_instruction_dataset(
                dataset,
                dataset_name=dataset_attr.dataset_name,
                dataset_format=dataset_attr.dataset_format,
                instruction_template=args.instruction_template,
                text_logger=text_logger,
            )

        train_dataset, eval_dataset = split_train_eval(
            dataset,
            do_eval=args.do_eval,
            eval_dataset_size=args.eval_dataset_size,
            max_eval_samples=args.max_eval_samples,
            do_train=args.do_train,
            max_train_samples=args.max_train_samples,
            text_logger=text_logger,
        )
        if train_dataset:
            train_datasets.append(train_dataset)
        if eval_dataset:
            eval_datasets.append(eval_dataset)

    concate_train = concatenate_datasets(
        train_datasets) if train_datasets else None
    concate_eval = concatenate_datasets(
        eval_datasets) if eval_datasets else None

    result_train = (
        f'Concatenated dataset list: {dataset_name_list}, #train dataset size: {len(concate_train)}'
        if concate_train else None)
    result_eval = (
        f'Concatenated dataset list: {dataset_name_list}, #eval dataset size: {len(concate_eval)}'
        if concate_eval else None)
    text_logger.info(result_train)
    text_logger.info(result_eval)
    return concate_train, concate_eval, mutliturn_lst[0]


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    args: argparse.Namespace,
    text_logger: logging.Logger,
) -> dict[str, torch.utils.data.Dataset]:
    train_dataset, eval_dataset, multi_turn = make_data_module(
        args, text_logger)
    max_seq_length = tokenizer.model_max_length
    dataset_cls = (VicunaDataset if args.conversation_template == 'vicnua' else
                   ConversationDataset)

    if not multi_turn:
        train_dataset = (SupervisedDataset(
            train_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length,
        ) if args.do_train else None)

        eval_dataset = (SupervisedDataset(
            eval_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length,
        ) if args.do_eval else None)

    else:
        train_dataset = (dataset_cls(
            train_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        ) if args.do_train else None)
        eval_dataset = (dataset_cls(
            eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        ) if args.do_eval else None)

    if args.do_train:
        train_info = f'train_dataset: {type(train_dataset)}, mutlti-turn: {multi_turn},  #length: {len(train_dataset)}'
        text_logger.info(train_info)

    if args.do_eval:
        eval_info = f'eval_dataset: {type(eval_dataset)}, mutlti-turn: {multi_turn}, #length: {len(eval_dataset)}'
        text_logger.info(eval_info)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, predict_with_generate=args.predict_with_generate)

    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator,
    }
