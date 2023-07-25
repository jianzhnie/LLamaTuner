import json
import sys

import numpy as np

sys.path.append('../')
from typing import Any, Dict

import transformers

from chatllms.data.conv_dataset import ConversationDataset, UltraChatDataset
from chatllms.data.data_utils import (DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN,
                                      DEFAULT_PAD_TOKEN, DEFAULT_UNK_TOKEN)

if __name__ == '__main__':
    # Load the raw data from the specified data_path
    data_path = '/home/robin/work_dir/llm/FastChat/data/dummy_conversation.json'
    with open(data_path, 'r') as file:
        raw_data = json.load(file)

    model_name_or_path = '/home/robin/checkpoints/baichuan7b'
    model_name_or_path = 'facebook/opt-125m'
    sources = [example['conversations'] for example in raw_data]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=64,
        padding_side='right',
        use_fast=False,
        add_special_tokens=False,
        tokenizer_type='llama' if 'llama' in model_name_or_path else 'gpt2',
    )
    # Define a dictionary to store any missing special tokens along with their default values
    special_tokens_dict: Dict[str, Any] = {}

    # Check if each special token is present. If not, add it to the special_tokens_dict with its default value.
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    print(tokenizer.bos_token)
    # # Apply the conversation function to the raw data
    dataset = ConversationDataset(sources, tokenizer, 64)

    for idx, data in enumerate(dataset):
        print('==' * 10)
        input_ids = data['input_ids']
        input_txt = tokenizer.decode(input_ids)
        print(input_txt)
        targets = data['labels']
        input_ids = np.array(input_ids)
        target_text = tokenizer.decode(targets)
        print(target_text)
        if idx > 10:
            break

    dataset = UltraChatDataset(sources, tokenizer, 128)
    for idx, data in enumerate(dataset):
        input_ids = data['input_ids']
        labels = data['labels']
        input_txt = tokenizer.decode(input_ids)
        target_text = tokenizer.decode(labels)
        print(input_txt)
        print(target_text)
        if idx > 10:
            break
