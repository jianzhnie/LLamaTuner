from typing import Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class PairwiseDataset(Dataset):
    """Dataset class for pairwise ranking tasks.

    Args:
        data_path: Path to the dataset.
        tokenizer: The tokenizer used to encode the input text.
        max_length: Maximum sequence length for the encoded inputs.
    """

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer,
                 split: str, max_length: int):

        self.pairs = self.create_comparison_dataset(data_path, split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError(
                f'Index {idx} out of range for TLDRDataset with length {len(self)}'
            )
        pair = self.pairs[idx]
        chosen_example, rejected_example = pair['chosen'], pair['rejected']

        chosen_encodings_dict = self.tokenizer(chosen_example,
                                               truncation=True,
                                               max_length=self.max_length,
                                               padding='max_length')
        rejected_encodings_dict = self.tokenizer(rejected_example,
                                                 truncation=True,
                                                 max_length=self.max_length,
                                                 padding='max_length')
        encodings_input = {}
        encodings_input['chosen_input_ids'] = chosen_encodings_dict[
            'input_ids']
        encodings_input['chosen_attention_mask'] = chosen_encodings_dict[
            'attention_mask']
        encodings_input['rejected_input_ids'] = rejected_encodings_dict[
            'input_ids']
        encodings_input['rejected_attention_mask'] = rejected_encodings_dict[
            'attention_mask']
        encodings_input['labels'] = 1.0

        encodings_input = {
            key: torch.tensor(val)
            for key, val in encodings_input.items()
        }

        return encodings_input

    def create_comparison_dataset(self, path: str, split: str = 'train'):
        dataset = load_dataset(path, split=split)
        pairs = []
        for prompt, chosen_summary, rejected_summary in zip(
                dataset['prompt'], dataset['chosen'], dataset['rejected']):
            pair = {}
            if chosen_summary == rejected_summary:
                continue
            if len(chosen_summary.split()) < 5 or len(
                    rejected_summary.split()) < 5:
                continue

            pair[
                'chosen'] = '<|startoftext|>' + prompt + '\n' + chosen_summary + '<|endoftext|>'
            pair[
                'rejected'] = '<|startoftext|>' + prompt + '\n' + rejected_summary + '<|endoftext|>'
            pairs.append(pair)

        return pairs
