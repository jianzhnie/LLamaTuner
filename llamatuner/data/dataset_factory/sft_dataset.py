import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from llamatuner.configs import DataArguments
from llamatuner.data.dataset_factory.dataset_utils import \
    _preprocess_visual_inputs
from llamatuner.data.template import Template
from llamatuner.utils.constants import IGNORE_INDEX, IMAGE_TOKEN
from llamatuner.utils.logger_utils import get_logger

logger = get_logger('llamatuner')


class SupervisedDataset2(Dataset):

    def __init__(
        self,
        examples: Dict[str, List[Any]],
        template: Template,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        data_args: DataArguments,
    ):
        super().__init__()
        self.examples = examples
        self.template = template
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args

        # Store valid indices to skip invalid examples
        self.valid_indices = self._get_valid_indices()

        if self.processor is not None:
            self.preprocess_visual_inputs = partial(_preprocess_visual_inputs,
                                                    processor=processor)

    def _get_valid_indices(self) -> List[int]:
        """Pre-check which examples are valid to avoid processing invalid ones later"""
        valid_idx = []
        for i in range(len(self.examples['prompt'])):
            if len(self.examples['prompt'][i]) % 2 == 1 and len(
                    self.examples['response'][i]) == 1:
                valid_idx.append(i)
            else:
                logger.warning('Dropped invalid example: {}'.format(
                    self.examples['prompt'][i] + self.examples['response'][i]))
        return valid_idx

    def _process_single_example(self, idx: int) -> Dict[str, Any]:
        """Process a single example on-demand"""
        example = {'input_ids': [], 'attention_mask': [], 'labels': []}

        # Get the actual index from valid indices
        actual_idx = self.valid_indices[idx]

        if self.processor is not None and not hasattr(self.processor,
                                                      'image_seq_length'):
            # Make a copy to avoid modifying the original data
            prompt_content = self.examples['prompt'][actual_idx][0]['content']
            if not prompt_content.startswith(IMAGE_TOKEN):
                prompt_content = IMAGE_TOKEN + prompt_content

        messages = self.examples['prompt'][actual_idx] + self.examples[
            'response'][actual_idx]
        input_ids, labels = [], []

        # Handle image tokens if needed
        if self.processor is not None and hasattr(self.processor,
                                                  'image_seq_length'):
            image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            input_ids += [image_token_id] * getattr(self.processor,
                                                    'image_seq_length')
            labels += [IGNORE_INDEX] * getattr(self.processor,
                                               'image_seq_length')

        # Process each turn in the conversation
        for turn_idx, (source_ids, target_ids) in enumerate(
                self.template.encode_multiturn(
                    self.tokenizer,
                    messages,
                    self.examples['system'][actual_idx],
                    self.examples['tools'][actual_idx],
                    self.data_args.cutoff_len,
                    self.data_args.reserved_label_len,
                )):
            if self.data_args.train_on_prompt:
                source_mask = source_ids
            elif turn_idx != 0 and self.template.efficient_eos:
                source_mask = [self.tokenizer.eos_token_id
                               ] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        example['input_ids'] = input_ids
        example['attention_mask'] = [1] * len(input_ids)
        example['labels'] = labels

        if self.processor is not None:
            example['pixel_values'] = self.preprocess_visual_inputs(
                self.examples['images'][actual_idx])

        return example

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._process_single_example(idx)


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
