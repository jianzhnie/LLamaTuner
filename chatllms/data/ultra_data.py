from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import LongTensor
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100


def data_collator(
        tokenizer: PreTrainedTokenizer,
        instances: Sequence[Dict[str,
                                 torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate and pad a batch of tokenized instances.

    Args:
        tokenizer: The tokenizer used to encode the data.
        instances: A list of tokenized instances, each a dict of input IDs and labels.

    Returns:
        A dict with the padded input IDs and labels along with the attention mask.
    """

    input_ids = [instance['input_ids'] for instance in instances]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    labels = [instance['labels'] for instance in instances]
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': input_ids.ne(tokenizer.pad_token_id),
    }


class PromptIterableDataset(IterableDataset):
    """
    Streaming iterable dataset for sequence-to-sequence style prompt learning.

    Args:
        raw_dataset: The original dataset of examples. Must have __iter__ and __len__ methods.
        sep: List of sentence separation tokens, e.g. ["EOS", "\n"].
        tokenizer: Tokenizer to use for encoding text.
        max_seq_length: Maximum sequence length for truncation.
        teacher_forcing: Whether to always feed ground truth tokens during training.
        truncate_method: How truncated sequences should be cut - 'head' or 'tail'.

    """
    def __init__(
        self,
        raw_dataset: Union[Dataset, List],
        sep: List[str] = ['EOS', '\n'],
        tokenizer: PreTrainedTokenizer = None,
        max_seq_length: Optional[int] = 512,
        teacher_forcing: bool = True,
        truncate_method: str = 'tail',
    ):
        assert hasattr(raw_dataset,
                       '__iter__'), 'Dataset must implement __iter__'
        assert hasattr(raw_dataset,
                       '__len__'), 'Dataset must implement __len__'

        self.raw_dataset = raw_dataset
        self.sep = sep
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.teacher_forcing = teacher_forcing
        self.truncate_method = truncate_method

        self._start_token = self.sep[-1]
        self._end_token = self.sep[
            0] if self.sep[0] != 'EOS' else self.tokenizer.eos_token

        assert self.truncate_method in [
            'head', 'tail'
        ], "Truncate method must be 'head' or 'tail'"
        assert self.teacher_forcing, 'Must use teacher forcing'

    @property
    def start_token(self) -> str:
        """Return unique start of sequence token"""
        return self._start_token

    @property
    def end_token(self) -> str:
        """Return unique end of sequence token"""
        return self._end_token

    def _tokenize(self, text: str) -> LongTensor:
        """Tokenize a single input text."""
        tokens = self.tokenizer(text, add_special_tokens=False)
        return LongTensor(tokens['input_ids'])

    def _truncate(self, sequence: LongTensor) -> LongTensor:
        """Truncate a sequence to the configured max length."""
        if len(sequence) > self.max_seq_length:
            if self.truncate_method == 'tail':
                sequence = sequence[:-(len(sequence) - self.max_seq_length)]
            elif self.truncate_method == 'head':
                sequence = sequence[-self.max_seq_length:]

        return sequence

    def _tokenize_example(self, example: dict) -> dict:
        """Tokenize a single example into input IDs and labels."""
        inputs, labels = [], []

        for i, text in enumerate(example['data']):

            # Alternate between user and assistant tokens
            speaker = 'User' if i % 2 == 0 else 'Assistant'

            # Tokenize text
            tokens = self._tokenize(f'{speaker}: {text} {self.end_token}')

            # Handle start token
            if i == 0:
                tokens = self._tokenize(self.tokenizer.bos_token + tokens)
            else:
                tokens = self._tokenize(self.start_token + tokens)

            # Add text tokens
            inputs.extend(tokens)

            # Add labels depending on teacher forcing
            if self.teacher_forcing:
                if i % 2 == 1:
                    # Add ground truth tokens for assistant
                    labels.extend(self._tokenize(text + self.end_token))
                else:
                    # Use ignore index for user input
                    labels.extend([IGNORE_INDEX] * len(tokens))
            else:
                labels.extend([IGNORE_INDEX] * len(tokens))

        return {'input_ids': LongTensor(inputs), 'labels': LongTensor(labels)}

    def __iter__(self):
        """Iterate through examples, tokenizing on the fly"""
        for example in self.raw_dataset:
            tokenized = self._tokenize_example(example)
            yield self._truncate(tokenized)

    def __len__(self) -> int:
        """Return length of the original dataset"""
        return len(self.raw_dataset)
