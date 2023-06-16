import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'

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


def local_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """
    Reads in a dataset from a file and returns it as a split train-test dataset.

    Args:
        dataset_name (str): The name of the dataset file to read in. The format is inferred based on the file extension.

    Returns:
        A tuple containing two datasets - the training subset and the testing subset.
    Raises:
        ValueError: If the specified file format is unsupported.

    """
    # Read in the full dataset from file based on the file format
    if dataset_name.endswith('.json'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(filename=dataset_name,
                                         format='jsonlines')
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(
            pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f'Unsupported dataset format: {dataset_name}')

    # Split the full dataset into train and test subsets
    split_dataset = full_dataset.train_test_split(test_size=0.1)

    return split_dataset


def load_data(dataset_name: str) -> Union[Dict[str, Dataset], None]:
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
        return load_dataset('tatsu-lab/alpaca')
    elif dataset_name == 'alpaca-clean':
        return load_dataset('yahma/alpaca-cleaned')
    elif dataset_name == 'chip2':
        return load_dataset('laion/OIG', data_files='unified_chip2.jsonl')
    elif dataset_name == 'self-instruct':
        return load_dataset('yizhongw/self_instruct', name='self_instruct')
    elif dataset_name == 'hh-rlhf':
        return load_dataset('Anthropic/hh-rlhf')
    elif dataset_name == 'longform':
        return load_dataset('akoksal/LongForm')
    elif dataset_name == 'oasst1':
        return load_dataset('timdettmers/openassistant-guanaco')
    elif dataset_name == 'vicuna':
        raise NotImplementedError('Vicuna data was not released.')
    elif dataset_name == 'dolly-15k':
        return load_dataset('databricks/databricks-dolly-15k')
    else:
        if os.path.exists(dataset_name):
            try:
                full_dataset = local_dataset(dataset_name)
                return full_dataset
            except:
                raise ValueError(f'Error loading dataset from {dataset_name}')
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


def format_dataset(dataset: Dataset,
                   dataset_name: str) -> Optional[Dict[str, Dataset]]:
    """
    Formats a given dataset based on its name and format.

    Args:
        dataset: A dataset object to be formatted.
        dataset_name: A string representing the name of the dataset to be formatted.

    Returns:
        A dictionary containing the formatted dataset if the dataset exists in the
        specified format.
        None if the dataset does not exist or if the format is not recognized.

    Examples:
        >>> format_dataset('alpaca')
        {'train': Dataset(...), 'validation': Dataset(...), 'test': Dataset(...)}

    """
    if dataset_name == 'alpaca' or dataset_name == 'alpaca-clean':
        dataset = dataset.map(extract_alpaca_dataset,
                              remove_columns=['instruction'])
    elif dataset_name == 'dolly-15k':
        dataset = dataset.rename_column('context', 'input')
        dataset = dataset.rename_column('response', 'output')
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
    elif os.path.exists(dataset_name):
        dataset = dataset.map(extract_alpaca_dataset,
                              remove_columns=['instruction'])
    else:
        return None  # invalid format

    # Remove unused columns.
    dataset = dataset.remove_columns([
        col for col in dataset.column_names['train']
        if col not in ['input', 'output']
    ])
    return dataset


def split_train_eval(
    dataset: Dataset,
    do_eval: bool = False,
    eval_dataset_size: float = 0.2,
    max_eval_samples: int = None,
    group_by_length: bool = False,
    do_train: bool = True,
    max_train_samples: int = None,
) -> Dict[str, Dataset]:
    """
    Prepare the training and evaluation datasets for a machine learning model.

    Args:
        dataset (DatasetDict): The complete dataset containing train, validation, and test splits.
        do_eval (bool, optional): Whether to use an evaluation dataset or not. Defaults to False.
        eval_dataset_size (float, optional): The size of the validation set if splitting from the training data.
            Ignored if `do_eval` is False. Defaults to 0.2.
        max_eval_samples (int, optional): The maximum number of samples to keep in the evaluation dataset.
            Ignored if `do_eval` is False or `None`. Defaults to None.
        group_by_length (bool, optional): Whether to group the data by length or not. Defaults to False.
        do_train (bool, optional): Whether to use a training dataset or not. Defaults to True.
        max_train_samples (int, optional): The maximum number of samples to keep in the training dataset.
            Ignored if `do_train` is False or `None`. Defaults to None.

    Returns:
        Dict[str, Dataset]: A dictionary containing the prepared training and evaluation datasets
        (if used), where the keys are 'train' and 'eval', respectively.
    """
    if not isinstance(dataset, DatasetDict):
        raise TypeError("The 'dataset' argument must be a DatasetDict object.")

    # Prepare evaluation dataset
    if do_eval:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            # Split train dataset in train and validation according to `eval_dataset_size`
            print(
                'Splitting train dataset in train and validation according to `eval_dataset_size`'
            )
            dataset = dataset['train'].train_test_split(
                test_size=eval_dataset_size, shuffle=True, seed=42)
            eval_dataset = dataset['test']

        # Reduce evaluation dataset size (if specified)
        if max_eval_samples is not None and len(
                eval_dataset) > max_eval_samples:
            eval_dataset = eval_dataset.select(np.arange(max_eval_samples))

        # Group data by length (if specified)
        if group_by_length:
            eval_dataset = eval_dataset.map(
                lambda x: {'length': len(x['input']) + len(x['output'])})

    # Prepare training dataset
    if do_train:
        train_dataset = dataset['train']

        # Reduce training dataset size (if specified)
        if max_train_samples is not None and len(
                train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(np.arange(max_train_samples))

        # Group data by length (if specified)
        if group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {'length': len(x['input']) + len(x['output'])})

    return {'train': train_dataset, 'eval': eval_dataset}


def make_data_module(args):
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    dataset = load_data(args.dataset_name)
    dataset = format_dataset(dataset, dataset_name=args.dataset_name)
    dataset_dict = split_train_eval(
        dataset,
        do_eval=args.do_eval,
        eval_dataset_size=args.eval_dataset_size,
        max_eval_samples=args.max_eval_samples,
        group_by_length=args.group_by_length,
        do_train=args.do_train,
        max_train_samples=args.max_train_samples,
    )

    return dataset_dict


@dataclass
class DataCollatorForCausalLM(object):
    """
    Data collator used for language modeling tasks. This collator takes in a sequence of examples
    (input/output pairs) and returns a dictionary containing the inputs and labels for training
    a causal language model.

    Parameters:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to tokenize the input and output text.
        source_max_len (int): The maximum length allowed for the input source text.
        target_max_len (int): The maximum length allowed for the target output text.
        train_on_source (bool): If True, the model will be trained on the source text. Otherwise, it will be trained
                                on both source and target text concatenated together.
        predict_with_generate (bool, default=False): If True, only the input_ids for the tokenized source text
                                                      are returned. This is useful during inference when generating
                                                      text sequences from the model.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        source_max_len: int,
        target_max_len: int,
        train_on_source: bool,
        predict_with_generate: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate

    def __call__(
            self, instances: Sequence[Dict[str,
                                           str]]) -> Dict[str, torch.Tensor]:
        """
        Takes a sequence of input/output pairs and returns a dictionary containing the inputs and labels
        for training a causal language model.

        Parameters:
            instances (Sequence[Dict[str, str]]): A sequence of input/output pairs. Each dictionary must contain
                                                  the keys 'input' and 'output'.

        Returns:
            data_dict (Dict[str, torch.Tensor]): A dictionary containing the input_ids, attention_mask,
                                                 and optionally the labels.
        """
        # Extract elements
        sources: List[str] = [
            f"{self.tokenizer.bos_token}{example['input']}"
            for example in instances
        ]
        targets: List[str] = [
            f"{example['output']}{self.tokenizer.eos_token}"
            for example in instances
        ]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']):
            if not self.predict_with_generate:
                input_ids.append(
                    torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    # train_on_source 默认设置为 False, 训练时不在 source  text 上计算损失
                    labels.append(
                        torch.tensor([
                            IGNORE_INDEX for _ in range(len(tokenized_source))
                        ] + copy.deepcopy(tokenized_target)))
                else:
                    # 如果 train_on_source 设置为 True, 训练时将 source text  和 target text 的标签合并, 然后计算损失
                    labels.append(
                        torch.tensor(
                            copy.deepcopy(tokenized_source +
                                          tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        ) if not self.predict_with_generate else None

        # Construct data dictionary containing inputs and labels
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict
