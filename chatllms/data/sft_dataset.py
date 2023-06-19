import copy
import logging
from dataclasses import dataclass
from typing import Dict, List

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from chatllms.data.data_utils import IGNORE_INDEX

logger = logging.getLogger(__name__)


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
        logging.warning('Loading data...')
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
    Collate examples for supervised fine-tuning.

    Args:
        tokenizer (PreTrainedTokenizer): The pre-trained tokenizer to use.
        predict_with_generate (bool): Whether to do prediction with generate or not.
    """

    tokenizer: PreTrainedTokenizer
    predict_with_generate: bool = False

    def __call__(
            self,
            instances: List[Dict[str,
                                 torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples for supervised fine-tuning on a sequence classification task \
            using a pre-trained tokenizer.

        Args:
            instances (List[Dict[str, torch.Tensor]]): A list of dictionaries containing the keys\
                  'input_ids' and 'labels'.

        Returns:
            A dictionary containing the collated batch with keys 'input_ids', 'labels', and 'attention_mask'.
        """

        # Extract input IDs and labels from each instance
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))

        # Pad sequences to be of equal length
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        ) if not self.predict_with_generate else None

        # Construct attention mask based on padded input IDs
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Return collated batch as dictionary
        data_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict
