from typing import Dict, List, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from chatllms.data.data_utils import IGNORE_INDEX as IGNORE_TOKEN_ID
from chatllms.data.utils.conversation import Conversation, SeparatorStyle


def extract_conversations_from_raw_data(sources):
    """Extracts conversations from raw data."""
    # Create a Conversation object
    conv = Conversation(
        name='vicuna_v1.1',
        system=
        'A chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=['USER', 'ASSISTANT'],
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=' ',
        sep2='</s>',
    )
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first message if it is not from the human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'  # noqa
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    return conversations


def preprocess(sources: Sequence[Dict[str, str]],
               tokenizer: PreTrainedTokenizer) -> Dict[str, List[int]]:
    """
    Preprocesses the data by tokenizing it.

    Args:
        sources (Sequence[Dict[str, str]]): List of conversation sources.
            Each source is a dictionary containing 'from' (sender role) and 'value' (message content).
        tokenizer (PreTrainedTokenizer): Tokenizer for tokenizing the conversations.

    Returns:
        Dict[str, List[int]]: A dictionary containing the preprocessed data.
            - 'input_ids': Tokenized input conversation IDs.
            - 'labels': Tokenized target conversation IDs.
            - 'attention_mask': Attention mask for the input conversation.

    Raises:
        AssertionError: If the roles in the conversation are not consistent.

    """

    # Create a Conversation object
    conv = Conversation(
        name='vicuna_v1.1',
        system=
        'A chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=['USER', 'ASSISTANT'],
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=' ',
        sep2='</s>',
    )
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first message if it is not from the human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for rou in rounds:
            if rou == '':
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # Ignore the user instructions
            target[cur_len:cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' (ignored)')

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

    Args:
        raw_data (List[Dict]): Raw input data.
        tokenizer (PreTrainedTokenizer): Tokenizer for preprocessing the data.
    """
    def __init__(self, raw_data: List[Dict[str, List[str]]],
                 tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()

        print('Formatting inputs...')

        # Extract conversations from raw_data
        sources = [example['conversations'] for example in raw_data]

        # Preprocess the input data using the provided tokenizer
        data_dict = preprocess(sources, tokenizer)

        # Assign preprocessed data to class attributes
        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']
        self.attention_mask = data_dict['attention_mask']

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset at the specified index.

        Args:
            index (int): Index of the example to retrieve.

        Returns:
            dict: Dictionary containing the input IDs, labels, and attention mask tensors.
        """
        return {
            'input_ids': self.input_ids[index],
            'labels': self.labels[index],
            'attention_mask': self.attention_mask[index],
        }


class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    """
    def __init__(
        self,
        raw_data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize the LazySupervisedDataset.

        Args:
            raw_data (List[Dict[str, str]]): The raw input data for the dataset.
            tokenizer (PreTrainedTokenizer): The pre-trained tokenizer instance.
        """
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.raw_data)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset at the given index.

        Args:
            i (int): The index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the preprocessed item with keys 'input_ids',
                                     'labels', and 'attention_mask'.
        """
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(
            [self.raw_data[i]['conversations']],
            tokenizer=self.tokenizer,
        )

        ret = {
            'input_ids': ret['input_ids'][0],
            'labels': ret['labels'][0],
            'attention_mask': ret['attention_mask'][0],
        }
        self.cached_data_dict[i] = ret

        return ret


class VicunaDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

    Args:
        raw_data (List[Dict]): Raw input data.
        tokenizer (PreTrainedTokenizer): Tokenizer for preprocessing the data.
    """
    def __init__(self, data_path) -> None:
        super(VicunaDataset, self).__init__()

        print('Formatting inputs...Skip in lazy mode')
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            self.raw_data = load_dataset('json', data_files=data_path)['train']
        else:
            self.raw_data = load_dataset(data_path)['train']

        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        ret = extract_conversations_from_raw_data(
            [self.raw_data[i]['conversations']])
        self.cached_data_dict[i] = ret
        return ret


def make_conversation_data_module(
    tokenizer: PreTrainedTokenizer,
    lazy_preprocess: bool,
    data_path: str,
) -> Dict[str, Dataset]:
    """
    Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer object.
        lazy_preprocess (bool): Flag indicating whether to use lazy preprocessing.
        data_path (str): The path to the data file or directory.

    Returns:
        dict: A dictionary containing the train_dataset and eval_dataset.

    """
    # Determine the appropriate dataset class based on lazy_preprocess flag

    dataset_cls = (LazySupervisedDataset
                   if lazy_preprocess else SupervisedDataset)

    print('Loading data...')
    # Load the raw data from the specified data_path
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        raw_data = load_dataset('json', data_files=data_path)['train']
    else:
        raw_data = load_dataset(data_path)['train']

    # Split the data into training and evaluation sets
    raw_data = raw_data.train_test_split(test_size=0.1)
    train_raw_data = raw_data['train']
    eval_raw_data = raw_data['test']

    print(f'#train {len(train_raw_data)}, #eval {len(eval_raw_data)}')

    # Create train and eval datasets using the chosen dataset class
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)

    print('train_dataset: ', train_dataset, type(train_dataset), 'length: ',
          len(train_dataset))
    print('eval_dataset: ', eval_dataset, type(eval_dataset), 'length: ',
          len(eval_dataset))
    return {'train_dataset': train_dataset, 'eval_dataset': eval_dataset}
