"""
Split long conversations based on certain max length.

Usage: python3 -m split_long_conversation.py \
    --in sharegpt_clean.json \
    --out sharegpt_split.json \
    --model-name-or-path $<model-name>

example:
python split_long_conversation.py \
    --in-file sharegpt_clean.json \
    --model-name-or-path  decapoda-research/llama-7b-hf
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import transformers
from clean_sharegpt import filter_invalid_roles, get_statistics, json_dump
from tqdm import tqdm


def make_sample(sample: Dict[str, any], start_idx: int,
                end_idx: int) -> Dict[str, any]:
    """
    Create a new sample dictionary by selecting conversations from the given sample.

    Args:
        sample (Dict[str, any]): The original sample dictionary.
        start_idx (int): The starting index of conversations to include.
        end_idx (int): The ending index of conversations to include.

    Returns:
        Dict[str, any]: The new sample dictionary with selected conversations.
    """
    assert (end_idx - start_idx) % 2 == 0
    conversations = sample['conversations'][start_idx:end_idx]
    return {
        'id': sample['id'] + '_' + str(start_idx),
        'conversations': conversations,
    }


def split_one_sample(sample: Dict[str, any]) -> List[Dict[str, any]]:
    """
    Split a single sample into multiple samples based on conversation lengths.

    Args:
        sample (Dict[str, any]): The original sample dictionary.
        max_length (int): The maximum length constraint for conversations.

    Returns:
        List[Dict[str, any]]: The list of new sample dictionaries.
    """
    tokenized_lens = []
    conversations = sample['conversations']

    # Truncate conversations to an even number of conversations
    conversations = conversations[:len(conversations) // 2 * 2]

    # Calculate the tokenized length for each conversation
    for conv in conversations:
        length = len(tokenizer(conv['value']).input_ids) + 6
        tokenized_lens.append(length)

    new_samples = []
    start_idx = 0  # The starting index of conversations to include
    cur_len = 0  # The current length of conversations included

    # Iterate through conversations and create new samples based on length constraints
    for end_idx in range(0, len(conversations), 2):
        round_len = tokenized_lens[end_idx] + tokenized_lens[end_idx + 1]
        if cur_len + round_len > max_length:
            sub_sample = make_sample(sample, start_idx, end_idx + 2)
            new_samples.append(sub_sample)
            start_idx = end_idx + 2
            cur_len = 0
        elif end_idx == len(conversations) - 2:
            sub_sample = make_sample(sample, start_idx, end_idx + 2)
            new_samples.append(sub_sample)
        cur_len += round_len

    return new_samples


def worker(input_data:List[Dict[str, Any]]):
    result = []
    for sample in input_data:
        result.extend(split_one_sample(sample))
    return result


def split_all(raw_data: List[Dict[str, Any]],
              tokenizer_: transformers.PreTrainedTokenizer,
              max_length_: int) -> List[Dict[str, Any]]:
    """
    Split the content into smaller parts based on the max token length constraint.

    Args:
        raw_data (List[Dict[str, Any]]): The list of samples to split.
        tokenizer (PreTrainedTokenizer): The tokenizer object used for tokenization.
        max_length (int): The maximum length allowed for each split.

    Returns:
        List[Dict[str, Any]]: The list of new sample dictionaries after splitting.
    """
    global tokenizer, max_length
    tokenizer = tokenizer_
    max_length = max_length_

    new_content = []

    # Split content into chunks
    chunks = [content[i : i + 1000] for i in range(0, len(content), 1000)]
    # Use tqdm to show progress bar during the execution
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(worker, raw_data),
                           desc='Splitting long conversations',
                           total=len(chunks)):
            new_content.extend(result)

    return new_content


def main(args):
    contents = json.load(open(args.in_file, 'r'))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='right',
        model_max_length=args.max_length,
        use_fast=False,
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,
    )
    print('Splitting long conversations...')
    split_data = split_all(contents, tokenizer, args.max_length)
    res1, res2 = get_statistics(split_data)
    # Save role_list_2 and role_res_2 to JSON files
    json_dump(res2, 'role_res_3.json')
    print(f'#in: {len(contents)}, #out: {len(split_data)}')
    print('Filtering invalid roles...')
    new_content = filter_invalid_roles(split_data)
    res1, res2 = get_statistics(new_content)
    # Save role_list_3 and role_res_3 to JSON files
    json_dump(res2, 'role_res_4.json')
    print(f'#in: {len(split_data)}, #out: {len(new_content)}')
    json_dump(new_content, args.out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, required=True)
    parser.add_argument('--out-file', type=str, default='sharegpt_split.json')
    parser.add_argument('--model-name-or-path', type=str, required=True)
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    main(args)
