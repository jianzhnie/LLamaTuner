"""Dataset for sequence-to-sequence response generation."""
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ConversationDataset(Dataset):
    """
    Dataset for multi-turn conversations.

    Args:
        conversations: List of conversation dictionaries with "human" and "assistant" turns.
        tokenizer: Tokenizer to encode input text.
        max_seq_length: Maximum sequence length for model inputs.
    """
    def __init__(self, conversations: List[Dict],
                 tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.roles = ['human', 'gpt']

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        self.examples = []
        for i, conversation in enumerate(conversations):
            dialog_context = []
            for j, turn in enumerate(conversation):
                assert turn['from'] == self.roles[j % 2]
                dialog_context.append(turn['value'])

            encoded_inputs = self.tokenizer(
                dialog_context,
                return_tensors='pt',
            )

            input_ids = [bos_token_id]
            target_mask = [0]

            for i, ids in enumerate(encoded_inputs.input_ids, start=1):
                input_ids += ids + [eos_token_id]
                if i % 2 == 0:
                    target_mask += [1] * (len(ids) + 1)
                else:
                    target_mask += [0] * (len(ids) + 1)

            assert len(input_ids) == len(target_mask)
            self.examples.append((input_ids, target_mask))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_ids, target_mask = self.examples[index]

        # Truncate sequences
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]

        # Create attention masks
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }


class ConversationDataCollator(object):
    """
    Collate and pad a batch of conversation examples to prepare for training.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: List[Dict[str,
                                           Any]]) -> Dict[str, torch.Tensor]:
        lengths = [len(ex['input_ids']) for ex in examples]
        max_length = min(max(lengths), self.max_seq_length)

        batch_input_ids = []
        batch_att_masks = []
        batch_target_masks = []

        for ex in examples:
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
