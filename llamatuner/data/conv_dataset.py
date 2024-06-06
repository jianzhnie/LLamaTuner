"""Dataset for sequence-to-sequence response generation."""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llamatuner.utils.constants import IGNORE_INDEX


@dataclass
class VicunaDataset(Dataset):
    """Dataset for multi-turn conversations using a Transformer model.

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
        """Initialize the dataset with conversations, tokenizer, and max
        sequence length.

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
        self.system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

        # Token to use at the start of each turn
        self.start_token = '\n'

    def tokenize_conversation(
            self,
            conversation: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a single conversation into input IDs and labels.

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
        """Get the prefix for a human turn.

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
        """Get the input IDs and labels for a specific conversation.

        Args:
            index: Index of the conversation

        Returns:
            Dictionary with input IDs and labels
        """
        conversation = self.raw_data[index]['conversations']
        input_ids, labels = self.tokenize_conversation(conversation)

        # Truncate sequence lengths
        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]

        return {'input_ids': input_ids, 'labels': labels}


@dataclass
class ConversationDataset(Dataset):
    """Dataset for multi-turn conversations using Transformer model.

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
        """Initialize the dataset with conversations, tokenizer and max
        sequence length."""
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.roles = ['human', 'gpt']

    def tokenize_conversation(
        self,
        conversation: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a single conversation into input IDs and labels.

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
        """Get the input IDs and labels for a specific conversation.

        Args:
            index: Index of the conversation

        Returns:
            Dictionary with input IDs and labels
        """
        conversation = self.raw_data[index]['conversations']
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
    """Collate and pad a batch of conversation examples to prepare for
    training."""

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
