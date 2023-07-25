"""Dataset for sequence-to-sequence response generation."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from chatllms.data.data_utils import IGNORE_INDEX


@dataclass
class UltraChatDataset(Dataset):
    """
    Dataset for multi-turn conversations using a Transformer model.

    Attributes:
        conversations: List of conversation dictionaries containing "human" and "assistant" turns
        tokenizer: Pretrained tokenizer to encode text
        max_seq_length: Maximum sequence length for model inputs
    """
    def __init__(self,
                 conversations: List[Dict],
                 tokenizer: PreTrainedTokenizer,
                 max_seq_length: int = 1024):
        """
        Initialize the dataset with conversations, tokenizer, and max sequence length.

        Args:
            conversations: List of conversation dictionaries containing "human" and "assistant" turns
            tokenizer: Pretrained tokenizer to encode text
            max_seq_length: Maximum sequence length for model inputs
        """
        self.conversations = conversations
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
        return len(self.conversations)

    def __getitem__(self, index: int) -> Dict:
        """
        Get the input IDs and labels for a specific conversation.

        Args:
            index: Index of the conversation

        Returns:
            Dictionary with input IDs and labels
        """
        conversation = self.conversations[index]
        input_ids, labels = self.tokenize_conversation(conversation)

        # Truncate sequence lengths
        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]

        return {'input_ids': input_ids, 'labels': labels}
