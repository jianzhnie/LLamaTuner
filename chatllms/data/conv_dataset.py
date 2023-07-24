"""Dataset for sequence-to-sequence response generation."""
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from chatllms.data.data_utils import IGNORE_INDEX
from chatllms.data.sft_dataset import DataCollatorForSupervisedDataset


@dataclass
class UltraChatDataset(Dataset):
    """
    Dataset for multi-turn conversations.

    Args:
        conversations: List of conversation dictionaries with "human" and "assistant" turns.
        tokenizer: Tokenizer to encode input text.
        max_seq_length: Maximum sequence length for model inputs.
    """
    def __init__(
        self,
        conversations: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.roles = ['human', 'gpt']
        self.system = "A chat between a curious user and an artificial intelligence assistant. \
            The assistant gives helpful, detailed, and polite answers to the user's questions."

        self.start_token = '\n'

        self.examples = []
        for i, conversation in enumerate(conversations):
            dialog_context = []
            for j, turn in enumerate(conversation):
                role = self.roles[j % 2]
                assert turn['from'] == role
                role_txt = turn['value']
                if j % 2 == 0:
                    if j == 0:
                        content = self.tokenizer.bos_token + self.system + role + ':' + role_txt + tokenizer.eos_token
                    else:
                        content = self.start_token + role + ':' + role_txt + tokenizer.eos_token
                else:
                    content = self.start_token + role + ':'

                dialog_context.append(content)

            encoded_inputs = self.tokenizer(
                dialog_context,
                add_special_tokens=False,
            )

            input_ids = [tokenizer.bos_token_id]
            target_mask = [0]
            targets = [IGNORE_INDEX]

            for i, ids in enumerate(encoded_inputs.input_ids, start=1):
                input_ids += ids + [tokenizer.eos_token_id]
                if i % 2 == 0:
                    target_mask += [1] * (len(ids) + 1)
                    targets += input_ids
                else:
                    target_mask += [0] * (len(ids) + 1)
                    targets += [IGNORE_INDEX] * (len(ids) + 1)

            assert len(input_ids) == len(target_mask)
            self.examples.append((input_ids, target_mask, targets))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_ids, target_mask, targets = self.examples[index]

        # Truncate sequences
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        targets = targets[:self.max_seq_length]

        # Create attention masks
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': input_ids,
            'labels': targets,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }


@dataclass
class ConversationDataset(Dataset):
    """
    Dataset for multi-turn conversations.

    Args:
        conversations: List of conversation dictionaries with "human" and "assistant" turns.
        tokenizer: Tokenizer to encode input text.
        max_seq_length: Maximum sequence length for model inputs.
    """
    def __init__(
        self,
        conversations: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.roles = ['human', 'gpt']

        self.examples = []
        for i, conversation in enumerate(conversations):
            dialog_context = []
            for j, turn in enumerate(conversation):
                role = self.roles[j % 2]
                assert turn['from'] == role
                dialog_context.append(turn['value'])

            encoded_inputs = self.tokenizer(
                dialog_context,
                add_special_tokens=False,
            )

            input_ids = [tokenizer.bos_token_id]
            target_mask = [0]
            targets = [IGNORE_INDEX]

            for i, ids in enumerate(encoded_inputs.input_ids):
                input_ids += ids + [tokenizer.eos_token_id]
                if i % 2 == 0:  # user
                    target_mask += [0] * (len(ids) + 1)
                    targets += [IGNORE_INDEX] * (len(ids) + 1)
                else:  # assistent
                    target_mask += [1] * (len(ids) + 1)
                    targets += input_ids

            assert len(input_ids) == len(target_mask)
            self.examples.append((input_ids, target_mask, targets))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        input_ids, target_mask, targets = self.examples[index]

        # Truncate sequences
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        targets = targets[:self.max_seq_length]

        # Create attention masks
        attention_mask = [1] * len(input_ids)

        data_dict = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(targets, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.long),
        }

        # data_dict = {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'target_mask': target_mask,
        #     'labels': targets,
        # }

        return data_dict


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
    data_path: str,
) -> Dict[str, Dataset]:
    """
    Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer object.
        lazy_preprocess (bool): Flag indicating whether to use lazy preprocessing.
        data_path (str): The path to the data file or directory.

    Returns:
        dict: A dictionary containing the train_dataset and eval_dataset.

    """
    # Determine the appropriate dataset class based on lazy_preprocess flag

    print('Loading data...')
    # Load the raw data from the specified data_path
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        raw_data = load_dataset('json', data_files=data_path)['train']
    else:
        raw_data = load_dataset(data_path)['train']

    # Split the data into training and evaluation sets
    raw_data = raw_data.train_test_split(test_size=0.1)
    train_raw_data = raw_data['train']
    eval_raw_data = raw_data['test']

    print(f'#train {len(train_raw_data)}, #eval {len(eval_raw_data)}')

    train_conversations = [x['conversations'] for x in train_raw_data]
    eval_conversations = [x['conversations'] for x in eval_raw_data]

    # Create train and eval datasets using the chosen dataset class
    train_dataset = ConversationDataset(train_conversations,
                                        tokenizer=tokenizer)
    eval_dataset = ConversationDataset(eval_conversations, tokenizer=tokenizer)

    print('train_dataset: ', train_dataset, type(train_dataset), 'length: ',
          len(train_dataset))
    print('eval_dataset: ', eval_dataset, type(eval_dataset), 'length: ',
          len(eval_dataset))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
