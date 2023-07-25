import copy
import logging
from dataclasses import dataclass
from typing import Dict, List

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from chatllms.data.data_utils import IGNORE_INDEX, make_data_module

logger = logging.getLogger(__name__)


class SFTInstructionDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

    Attributes:
        PROMPT_DICT (dict): A dictionary containing prompts for the model to complete.

    Methods:
        __init__(self, data_path: str, tokenizer: PreTrainedTokenizer): Initializes a SupervisedDataset object.
        __len__(self) -> int: Returns the length of the dataset.
        __getitem__(self, idx) -> Dict[str, torch.Tensor]: Retrieves an example from the dataset at the specified index.

    """
    def __init__(
        self,
        raw_data: datasets.DatasetDict,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 1024,
    ):
        """
        Initializes a SupervisedDataset object.

        Args:
            data_path (str): The path to the training data file.
            tokenizer (PreTrainedTokenizer): The tokenizer object used to tokenize the input examples.

        """
        super(SFTInstructionDataset, self).__init__()
        # Load the dataset and format it
        self.dataset = raw_data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of examples in the dataset.

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves an example from the dataset at the specified index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the input_ids, labels, input_len, source_input_ids, and
            source_len tensors.

        """
        example = self.dataset[idx]
        # Tokenize the source text
        src_txt = example['input']
        src_txt = f'{self.tokenizer.bos_token}{src_txt}{self.tokenizer.bos_token}'
        tokenized_src = self.tokenizer(
            src_txt,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )
        tgt_txt = example['output']
        tgt_txt = f'{tgt_txt}{self.tokenizer.eos_token}'
        # Tokenize the example and source text
        tokenized_tgt = self.tokenizer(
            tgt_txt,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )
        src_ids = tokenized_src['input_ids']
        tgt_ids = tokenized_tgt['input_ids']

        # Extract the input_ids tensor
        input_ids = torch.tensor(src_ids + tgt_ids)
        # Create the labels tensor
        labels = input_ids.clone()
        labels[:len(src_ids)] = IGNORE_INDEX

        # Construct data dictionary containing inputs and labels
        data_dict = {'input_ids': input_ids, 'labels': labels}

        return data_dict


@dataclass
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

        Args:
            hf_dataset (dataset): The preprocesed dataset to load.
            tokenizer (PreTrainedTokenizer): The tokenizer to use when tokenizing the data.
            source_max_len (int): The maximum length allowed for the source text.
            target_max_len (int): The maximum length allowed for the target text.
            train_on_source (bool): If True, the model will be trained on the source text as well as the target text.
            predict_with_generate (bool): If True, the model will generate predictions instead of training.
    """
    def __init__(
        self,
        hf_dataset: datasets.DatasetDict,
        tokenizer: PreTrainedTokenizer,
        source_max_len: int,
        target_max_len: int,
        train_on_source: bool,
        predict_with_generate: bool = False,
    ):

        super(SupervisedDataset, self).__init__()
        # Load the dataset and format it
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return an item from the dataset based on its index."""
        example = self.dataset[idx]
        # Tokenize the source text
        source_txt = f"{self.tokenizer.bos_token}{example['input']}"
        tokenized_source = self.tokenizer(
            source_txt,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Tokenize the target text
        target_txt = f"{example['output']}{self.tokenizer.eos_token}"
        tokenized_target = self.tokenizer(
            target_txt,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        src_ids = tokenized_source['input_ids']
        tgt_ids = tokenized_target['input_ids']
        if not self.predict_with_generate:
            # If not generating predictions, concatenate the input and target ids
            input_ids = torch.tensor(src_ids + tgt_ids)
            if not self.train_on_source:
                # If not training on the source text, set the labels to IGNORE_INDEX \
                # for the input ids and the target ids
                labels = torch.tensor(
                    [IGNORE_INDEX
                     for _ in range(len(src_ids))] + copy.deepcopy(tgt_ids))
            else:
                # If training on the source text, set the labels to the concatenated \
                # input and target ids
                labels = torch.tensor(copy.deepcopy(src_ids + tgt_ids))
        else:
            # If generating predictions, only use the source ids as input
            input_ids = torch.tensor(src_ids)
            labels = None

        # Construct data dictionary containing inputs and labels
        data_dict = {'input_ids': input_ids, 'labels': labels}

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """
    Collate and pad examples for supervised training.
    """

    tokenizer: PreTrainedTokenizer
    predict_with_generate: bool = False

    def __call__(
            self,
            examples: List[Dict[str,
                                torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into dictionary for supervised training.

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


def make_instruction_data_module(tokenizer: PreTrainedTokenizer, args):
    train_dataset, eval_dataset = make_data_module(args)
    train_dataset = SupervisedDataset(
        train_dataset,
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    ) if args.do_train else None

    eval_dataset = SupervisedDataset(
        eval_dataset,
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    ) if args.do_eval else None

    print(
        f'train_dataset: {type(train_dataset)}, #length: {len(train_dataset)}'
    ) if args.do_train else None
    print(f'eval_dataset: {type(eval_dataset)}, #length: {len(eval_dataset)}'
          ) if args.do_eval else None
    print('Adding data collator: ', DataCollatorForSupervisedDataset)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, predict_with_generate=args.predict_with_generate)

    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
