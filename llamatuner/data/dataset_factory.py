import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import datasets
import torch
from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llamatuner.utils.constants import IGNORE_INDEX
from llamatuner.utils.logger_utils import get_logger

logger = get_logger('llamatuner')


@dataclass
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning of instruction following models.

    Converts raw dataset containing source/target instructions
    into tokenized input/target pairs with truncation and padding.

    Attributes:
        dataset: The raw dataset containing source/target examples
        tokenizer: Tokenizer to use for encoding text
        max_seq_len: Maximum sequence length for truncation
    """

    def __init__(
        self,
        raw_data: DatasetDict,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 1024,
        train_on_source: bool = False,
        predict_with_generate: bool = False,
    ):
        """Initialize the dataset with the raw data and tokenizer.

        Args:
            raw_data: Raw dataset containing source/target examples
            tokenizer: Tokenizer to encode text
            max_seq_len: Max sequence length for truncation
            train_on_source (bool): If True, the model will be trained on the source text as well as the target text.
            predict_with_generate (bool): If True, the model will generate predictions instead of training.
        """
        super(SupervisedDataset, self).__init__()

        self.dataset = raw_data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Convert an raw example into tokenized input/target pair.

        Args:
            idx: Index of the example in the dataset

        Returns:
            input_ids: tokenized input sequence
            labels: tokenized target sequence
        """

        example = self.dataset[idx]
        source_text = f"{self.tokenizer.bos_token}{example['input']}"
        target_text = f"{example['output']}{self.tokenizer.eos_token}"

        # Tokenize the source text
        tokenized_source = self.tokenizer(source_text,
                                          max_length=self.max_seq_len,
                                          truncation=True,
                                          add_special_tokens=False)
        # Tokenize the target text
        tokenized_target = self.tokenizer(target_text,
                                          max_length=self.max_seq_len,
                                          truncation=True,
                                          add_special_tokens=False)

        source_ids = tokenized_source['input_ids']
        target_ids = tokenized_target['input_ids']

        # Extract the input_ids tensor
        if len(source_ids) > self.max_seq_len:
            logger.info(
                f'Source length {len(source_ids)} exceeds max seq length of {self.max_seq_len}'
            )
        # Create the labels tensor
        if len(target_ids) > self.max_seq_len:
            logger.info(
                f'Target length {len(target_ids)} exceeds max seq length of {self.max_seq_len}'
            )
        if not self.predict_with_generate:
            # If not generating predictions, concatenate the input and target ids
            input_ids = torch.tensor(source_ids + target_ids)
            if not self.train_on_source:
                # If not training on the source text, set the labels to IGNORE_INDEX \
                # for the input ids and the target ids
                labels = torch.tensor(
                    [IGNORE_INDEX for _ in range(len(source_ids))] +
                    copy.deepcopy(target_ids))
            else:
                # If training on the source text, set the labels to the concatenated \
                # input and target ids
                labels = torch.tensor(copy.deepcopy(source_ids + target_ids))
        else:
            # If generating predictions, only use the source ids as input
            input_ids = torch.tensor(source_ids)
            labels = None

        # Construct data dictionary containing inputs and labels
        data_dict = {'input_ids': input_ids, 'labels': labels}

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate and pad examples for supervised training."""

    tokenizer: PreTrainedTokenizer
    predict_with_generate: bool = False

    def __call__(
            self,
            examples: List[Dict[str,
                                torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate examples into dictionary for supervised training.

        Args:
            examples: List of examples, each containing 'input_ids' and 'labels'

        Returns:
            Dictionary with padded 'input_ids', 'attention_mask' and optionally 'labels'
        """

        # Extract input_ids and labels
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]

        # Pad input sequences
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)

        # Pad labels if needed
        if not self.predict_with_generate:
            labels = pad_sequence(labels,
                                  batch_first=True,
                                  padding_value=IGNORE_INDEX)

        # Create attention mask based on padded input
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Assemble final dict
        data_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict


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
