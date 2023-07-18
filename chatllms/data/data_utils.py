import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from .data_maps import get_dataset_path

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

PROMPT_DICT = {
    'prompt_input': ('{instruction}\n\n{input}\n\n'),
    'prompt_no_input': ('{instruction}\n\n'),
}

PROMPTS_WITH_INPUT = [
    # input encoding template, output encoding template, weight
    ('{instruction}\n\n{input}\n\n', 0.2),
    ('{instruction}\n{input}\n\n', 0.1),
    ('{instruction}\n{input}\n', 0.1),
    ('{instruction}\n\nInput: {input}\n\nOutput:', 0.05),
    ('{instruction}\nInput: {input}\nOutput:', 0.05),
    ('{instruction}\n{input}\n\nResponse:', 0.05),
    ('{instruction}\n\nAdditional Context:\n{input}\n\nAnswer:', 0.05),
    ('Task: {instruction}\nInput: {input}\nOutput:', 0.05),
    ('Task: {instruction}\n\n{input}\n\n', 0.05),
    ('Task: {instruction}\n\n{input}\n\nAnswer:', 0.05),
    ('You need to complete the following task:\n\n{instruction}\n\n{input}\n\nAnswer:',
     0.05),
    ('{instruction}\n\nNow complete the following instance -\nInput: {input}\nOutput:',
     0.05),
    ('Instruction:{instruction}\n\nInput: {input}\n\n', 0.05),
    ('Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: ',
     0.1),  # alpaca template
]

PROMPTS_WITHOUT_INPUT = [
    ('{instruction}\n\n', 0.2),
    ('{instruction}\n', 0.1),
    ('{instruction}\n\nOutput:', 0.1),
    ('{instruction}\nOutput:', 0.05),
    ('{instruction}\nResponse:', 0.05),
    ('{instruction}\n\nAnswer:', 0.05),
    ('Task: {instruction}\n\n', 0.05),
    ('Instruction: {instruction}\n', 0.05),
    ('Instruction: {instruction}\nOutput:', 0.05),
    ('You need to complete the following task:\n\n{instruction}\n\n', 0.05),
    ('Can you help with this?\n\n{instruction}\n', 0.05),
    ('Plase answer the following request: {instruction}\nAnswer:', 0.05),
    ('Tell me how would you respond to the following request.\n{instruction}\n',
     0.05),
    ('Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:',
     0.1),  # alpaca template
]


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


def extract_vicuna_dataset(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts the input and output portions of a single conversation example from the Vicuña format.

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


def extract_instruction_dataset(
    example: Dict[str, Any],
    random_template: bool = True,
) -> Dict[str, str]:
    if random_template:
        # Randomly choose prompt template
        if example.get('input', '') != '':
            # Input exists, choose from prompts with input
            prompt_template, _ = random.choices(
                PROMPTS_WITH_INPUT,
                weights=[w for _, w in PROMPTS_WITH_INPUT])[0]
        else:
            # No input, choose from prompts without input
            prompt_template, _ = random.choices(
                PROMPTS_WITHOUT_INPUT,
                weights=[w for _, w in PROMPTS_WITHOUT_INPUT])[0]

    else:
        # Not random, use pre-defined templates
        if example.get('input', '') != '':
            prompt_template = PROMPT_DICT['prompt_input']
        else:
            prompt_template = PROMPT_DICT['prompt_no_input']

    # Format prompt with example
    formatted_prompt = prompt_template.format(**example)

    return {'input': formatted_prompt}


def local_dataset(dataset_path: str,
                  eval_dataset_size: float = 0.1) -> Tuple[Dataset, Dataset]:
    """
    Reads in a dataset from a file and returns it as a split train-test dataset.

    Args:
        dataset_path (str): The name of the dataset file to read in. \
            The format is inferred based on the file extension.

    Returns:
        A tuple containing two datasets - the training subset and the testing subset.
    Raises:
        ValueError: If the specified file format is unsupported.

    """

    # Read in the full dataset from file based on the file format
    if dataset_path.endswith('.json'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_path)
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


def load_data(dataset_name: str,
              dataset_path: str) -> Union[Dict[str, Dataset], None]:
    """
    Load a dataset based on its name.

    Args:
        dataset_name: A string representing the name of the dataset to be loaded.
        dataset_path: A string representing the path to the dataset to be loaded.

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
        dataset = load_dataset(dataset_path,
                               cache_dir='~/.cache/huggingface/datasets')
        return dataset
    else:
        # Load dataset from local file
        try:
            full_dataset = local_dataset(dataset_path)
            return full_dataset
        except:
            raise ValueError(f'Error loading dataset from {dataset_path}')


def format_dataset(
        dataset: Dataset,
        dataset_name: str,
        prompt_template: str = 'instruction') -> Optional[Dict[str, Dataset]]:
    """
    Formats a given dataset based on its name and format.


    Removes unused columns, renames columns to 'input' and 'output', 
    and applies dataset-specific formatting based on the dataset_name.

    Returns formatted dataset dict if dataset can be formatted, else None.

    Args:
        dataset: A dataset object to be formatted.
        dataset_name: A string representing the name of the dataset to be formatted.

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
        """Format Self-Instruct dataset."""
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

    def _format_vicuna(dataset: Dataset) -> Dataset:
        """Format Vicuna dataset."""
        dataset = dataset.map(extract_vicuna_dataset)
        return dataset

    def _remove_unused_columns(dataset):
        """Remove columns not named 'input' or 'output'."""
        dataset = dataset.remove_columns([
            col for col in dataset.column_names['train']
            if col not in ['input', 'output']
        ])
        return dataset

    print("formate the dataset in structure")
    if dataset_name == 'dolly-15k':
        dataset = _format_dolly15k(dataset)
    elif dataset_name == 'chip2':
        dataset = _format_chip2(dataset)
    elif dataset_name == 'self-instruct':
        dataset = _format_self_instruct(dataset)
    elif dataset_name == 'hh-rlhf':
        dataset = _format_hh_rlhf(dataset)
    elif dataset_name == 'oasst1':
        dataset = _format_oasst1(dataset)
    elif dataset_name == 'vicuna':
        dataset = _format_vicuna(dataset)
    elif dataset_name == 'sharegpt':
        raise NotImplementedError
    else:
        pass

    # encode_instruction_example
    print(f"Encoding the instruction example refer to : {prompt_template} ", )
    if prompt_template == 'alpaca':
        dataset = dataset.map(extract_alpaca_dataset)
    elif prompt_template == 'instruction':
        dataset = dataset.map(extract_instruction_dataset)

    # Remove unused columns.
    print("Removing the unused columns.")
    dataset = _remove_unused_columns(dataset)
    return dataset


def split_train_eval(
    dataset: Dataset,
    do_eval: bool = False,
    eval_dataset_size: float = 0.1,
    max_eval_samples: int = None,
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
        do_train (bool, optional): Whether to use a training dataset or not. Defaults to True.
        max_train_samples (int, optional): The maximum number of samples to keep in the training dataset.
            Ignored if `do_train` is False or `None`. Defaults to None.

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

    # Prepare training dataset
    if do_train:
        train_dataset = dataset['train']

        # Reduce training dataset size (if specified)
        if max_train_samples is not None and len(
                train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(np.arange(max_train_samples))

    return train_dataset, eval_dataset


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
    train_datasets = []
    eval_datasets = []
    dataset_name_list = args.dataset_name.split(',')
    print(f'Loading datasets: {dataset_name_list}')
    for dataset_name in dataset_name_list:
        dataset_path = get_dataset_path(dataset_name,
                                        data_dir=args.data_dir,
                                        load_from_local=args.load_from_local)

        dataset = load_data(dataset_name, dataset_path)
        dataset = format_dataset(dataset, dataset_name=dataset_name)

        train_dataset, eval_dataset = split_train_eval(
            dataset,
            do_eval=args.do_eval,
            eval_dataset_size=args.eval_dataset_size,
            max_eval_samples=args.max_eval_samples,
            do_train=args.do_train,
            max_train_samples=args.max_train_samples,
        )
        if train_dataset:
            print('=' * 80)
            print('loaded dataset:', dataset_name, 'train data size:',
                  len(train_dataset))
            train_datasets.append(train_dataset)
        if eval_dataset:
            print('=' * 80)
            print('loaded dataset:', dataset_name, 'eval data size:',
                  len(eval_dataset))
            eval_datasets.append(eval_dataset)

    print('=' * 80)
    concate_train = concatenate_datasets(
        train_datasets) if train_datasets else None
    print(f'Concatenate train dataset size: {len(concate_train)}'
          ) if concate_train else None
    concate_eval = concatenate_datasets(
        eval_datasets) if eval_datasets else None
    print(f'Concatenate eval dataset size: {len(concate_eval)}'
          ) if concate_eval else None
    return concate_train, concate_eval


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
