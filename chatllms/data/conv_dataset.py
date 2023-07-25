"""Dataset for sequence-to-sequence response generation."""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from chatllms.data.data_utils import IGNORE_INDEX
from chatllms.data.sft_dataset import DataCollatorForSupervisedDataset


@dataclass
class VicunaDataset(Dataset):
    """
    Dataset for multi-turn conversations using a Transformer model.

    Attributes:
        raw_data: The preprocessed dataset dict to load
        tokenizer: Pretrained tokenizer to encode text
        max_seq_length: Maximum sequence length for model inputs
    """
    def __init__(
        self,
        raw_data: datasets.DatasetDict,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
    ):
        """
        Initialize the dataset with conversations, tokenizer, and max sequence length.

        Args:
            raw_data: The preprocessed dataset dict to load
            tokenizer: Pretrained tokenizer to encode text
            max_seq_length: Maximum sequence length for model inputs
        """
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Mapping from speaker to role
        self.roles = {'human': 'USER', 'gpt': 'ASSISTANT'}

        # Description of the conversation
        self.system = 'A friendly conversation between a human and an artificial intelligence assistant.'

        # Token to use at the start of each turn
        self.start_token = '\n'

    def tokenize_conversation(
            self,
            conversation: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a single conversation into input IDs and labels.

        Args:
            conversation: List of turns in the conversation

        Returns:
            input_ids: Tensor of input IDs
            labels: Tensor of word IDs for language modeling
        """

        # Arrays to store token IDs for input and labels
        input_ids = []
        labels = []

        # Track speaker roles
        roles = ['USER', 'ASSISTANT']

        # Tokenize each turn in the conversation
        for i, turn in enumerate(conversation):
            role = self.roles[turn['from']]
            assert role == roles[i % 2], f'{i}'

            # Get turn text
            text = turn['value']

            # For human turn, tokenize prompt
            if i % 2 == 0:
                prefix = self._get_human_prefix(i, role)
                prompt = prefix + text + self.tokenizer.eos_token
                tokenized = self.tokenizer(prompt, add_special_tokens=False)
                input_ids += tokenized['input_ids']
                labels += [IGNORE_INDEX] * len(tokenized['input_ids'])

            # For assistant turn, tokenize response
            else:
                prefix = self.start_token + role + ': '
                tokenized_prefix = self.tokenizer(prefix,
                                                  add_special_tokens=False)
                input_ids += tokenized_prefix['input_ids']
                labels += [IGNORE_INDEX] * len(tokenized_prefix['input_ids'])

                response = text + self.tokenizer.eos_token
                tokenized_response = self.tokenizer(response,
                                                    add_special_tokens=False)
                input_ids += tokenized_response['input_ids']
                labels += tokenized_response['input_ids']

        assert len(input_ids) == len(
            labels), f'{len(input_ids)} != {len(labels)}'

        return torch.tensor(input_ids), torch.tensor(labels)

    def _get_human_prefix(self, turn_id: int, role: str) -> str:
        """
        Get the prefix for a human turn.

        Args:
            turn_id: Index of the current turn
            role: Current speaker role

        Returns:
            prefix: Prefix string including special tokens
        """
        if turn_id == 0:
            prefix = self.tokenizer.bos_token + self.system + role + ': '
        else:
            prefix = self.start_token + role + ': '
        return prefix

    def __len__(self) -> int:
        """Get the number of conversations."""
        return len(self.raw_data)

    def __getitem__(self, index: int) -> Dict:
        """
        Get the input IDs and labels for a specific conversation.

        Args:
            index: Index of the conversation

        Returns:
            Dictionary with input IDs and labels
        """
        conversation = self.raw_data[index]['conversation']
        input_ids, labels = self.tokenize_conversation(conversation)

        # Truncate sequence lengths
        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]

        return {'input_ids': input_ids, 'labels': labels}


@dataclass
class ConversationDataset(Dataset):
    """
    Dataset for multi-turn conversations using Transformer model.

    Attributes:
        raw_data: The preprocessed dataset dict to load
        tokenizer: Pretrained tokenizer
        max_seq_length: Maximum length of sequence
    """
    def __init__(
        self,
        raw_data: datasets.DatasetDict,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
    ):
        """
        Initialize the dataset with conversations, tokenizer and max sequence length.
        """
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.roles = ['human', 'gpt']

    def tokenize_conversation(
        self,
        conversation: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize a single conversation into input IDs and labels.

        Args:
            conversation: List of turns in the conversation

        Returns:
            input_ids: Tensor of input IDs
            labels: Tensor of word IDs for language modeling
        """

        context = []
        for i, turn in enumerate(conversation):
            role = turn['from']
            assert role == self.roles[i % 2]
            context.append(turn['value'])

        encoded = self.tokenizer(context, add_special_tokens=False)

        input_ids = [self.tokenizer.bos_token_id]
        target_mask = [0]
        labels = [IGNORE_INDEX]

        for i, ids in enumerate(encoded.input_ids):
            input_ids += ids + [self.tokenizer.eos_token_id]

            if i % 2 == 0:  # Human turn
                target_mask += [0] * (len(ids) + 1)
                labels += [IGNORE_INDEX] * (len(ids) + 1)

            else:  # Assistant turn
                target_mask += [1] * (len(ids) + 1)
                labels += ids + [self.tokenizer.eos_token_id]

        assert len(input_ids) == len(target_mask) == len(labels)

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_mask, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get the input IDs and labels for a specific conversation.

        Args:
            index: Index of the conversation

        Returns:
            Dictionary with input IDs and labels
        """
        conversation = self.raw_data[index]['conversation']
        input_ids, target_mask, labels = self.tokenize_conversation(
            conversation)

        # Truncate sequence
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        labels = labels[:self.max_seq_length]

        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'target_mask': target_mask
        }


@dataclass
class ConversationDataCollator(object):
    """
    Collate and pad a batch of conversation examples to prepare for training.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        lengths = [len(ex['input_ids']) for ex in batch]
        max_length = min(max(lengths), self.max_seq_length)

        batch_input_ids = []
        batch_att_masks = []
        batch_target_masks = []

        for ex in batch:
            print(ex)
            input_ids = ex['input_ids']
            attention_mask = ex['attention_mask']
            target_mask = ex['target_mask']

            padding_length = max_length - len(input_ids)

            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            target_mask = target_mask + [0] * padding_length

            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            batch_input_ids.append(input_ids)
            batch_att_masks.append(attention_mask)
            batch_target_masks.append(target_mask)

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_att_masks = torch.tensor(batch_att_masks, dtype=torch.long)
        batch_target_masks = torch.tensor(batch_target_masks, dtype=torch.long)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_att_masks,
            'target_mask': batch_target_masks
        }


def make_conversation_data_module(
    tokenizer: PreTrainedTokenizer,
    use_vicuna_prompt: bool = False,
    data_path: str = './data/share_gpt.json',
    test_size: float = 0.1,
) -> Dict[str, Dataset]:
    """
    Create dataset and collator for conversation modeling.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer object.
        use_vicuna_prompt (bool): Flag indicating whether to use vicuna_prompt.
        data_path (str): The path to the data file or directory.

    Returns:
        dict: A dictionary containing the train_dataset and eval_dataset.

    """
    # Determine the appropriate dataset class based on dataset_type flag
    dataset_cls = (VicunaDataset if use_vicuna_prompt else ConversationDataset)

    print('Loading data...')
    # Load the raw data from the specified data_path
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        raw_data = load_dataset('json', data_files=data_path)['train']
    else:
        raw_data = load_dataset(data_path)['train']

    # Map conversations to dict format
    raw_data = raw_data.map(lambda x: {'conversations': x['conversations']})
    # Split the data into training and evaluation sets
    raw_data = raw_data.train_test_split(test_size=test_size)
    train_raw_data = raw_data['train']
    eval_raw_data = raw_data['test']

    print(f'#train {len(train_raw_data)}, #eval {len(eval_raw_data)}')

    # Create train and eval datasets using the chosen dataset class
    max_length = tokenizer.model_max_length
    train_dataset = dataset_cls(train_raw_data,
                                tokenizer=tokenizer,
                                max_seq_length=max_length)
    eval_dataset = dataset_cls(train_raw_data,
                               tokenizer=tokenizer,
                               max_seq_length=max_length)

    print('train_dataset: ', train_dataset, type(train_dataset), 'length: ',
          len(train_dataset))
    print('eval_dataset: ', eval_dataset, type(eval_dataset), 'length: ',
          len(eval_dataset))

    # Create data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print('data_collator: ', data_collator, type(data_collator))

    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
