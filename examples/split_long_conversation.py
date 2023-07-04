"""
Split long conversations based on certain max length.

Usage: python3 -m split_long_conversation.py \
    --in sharegpt_clean.json \
    --out sharegpt_split.json \
    --model-name-or-path $<model-name>

example: 
python split_long_conversation.py \
    --in-file sharegpt_formate_role_filter.json \
    --model-name-or-path  decapoda-research/llama-7b-hf
"""
import argparse
import json
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor
import transformers
from tqdm import tqdm
from typing import List, Dict


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
    return {
        "id": sample["id"] + "_" + str(start_idx),
        "model": sample.get("model", ""),
        "conversations": sample["conversations"][start_idx:end_idx],
    }


def split_one_sample(sample: Dict[str, any]) -> List[Dict[str, any]]:
    """
    Split a single sample into multiple samples based on conversation lengths.

    Args:
        sample (Dict[str, any]): The original sample dictionary.

    Returns:
        List[Dict[str, any]]: The list of new sample dictionaries.
    """
    tokenized_lens = []
    conversations = sample["conversations"]

    # Truncate conversations to an even number of conversations
    conversations = conversations[:len(conversations) // 2 * 2]

    # Calculate the tokenized length for each conversation
    for c in conversations:
        length = len(tokenizer(c["value"]).input_ids) + 6
        tokenized_lens.append(length)

    if len(conversations) % 2 != 0 or len(conversations) < 2:
        return []

    new_samples = []
    start_idx = 0  # The starting index of conversations to include
    cur_len = 0  # The current length of conversations included
    # Iterate through conversations and create new samples based on length constraints
    for i in range(0, len(conversations), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1]
        if cur_len + tmp_len > max_length:
            new_samples.append(make_sample(sample, start_idx, i))
            start_idx = i
            cur_len = 0
        elif i == len(conversations) - 2:
            new_samples.append(make_sample(sample, start_idx, i + 2))

        cur_len += tmp_len

    return new_samples


def split_all(contents: List[Dict[str, Any]],
              tokenizer_: transformers.PreTrainedTokenizer,
              max_length_: int) -> List[Dict[str, Any]]:
    """
    Split the content into smaller parts based on the max token length constraint.

    Args:
        contents (List[Dict[str, Any]]): The list of samples to split.
        tokenizer (PreTrainedTokenizer): The tokenizer object used for tokenization.
        max_length (int): The maximum length allowed for each split.

    Returns:
        List[Dict[str, Any]]: The list of new sample dictionaries after splitting.
    """
    global tokenizer, max_length
    tokenizer = tokenizer_
    max_length = max_length_

    new_content = []

    # Use tqdm to show progress bar during the execution
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(split_one_sample, contents),
                           desc="Splitting long conversations",
                           total=len(contents)):
            new_content.extend(result)

    return new_content


def filter_invalid_roles(
        contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out invalid contents based on the roles assigned to each conversation.

    Args:
        contents (List[Dict[str, Any]]): The list of contents to filter.

    Returns:
        List[Dict[str, Any]]: The list of valid contents after filtering.
    """
    new_content = []

    for i, content in enumerate(contents):
        roles = ["human", "gpt"]

        # Skip empty conversations
        if len(content["conversations"]) <= 0:
            continue

        valid = True

        # Check the roles assigned to each conversation
        for j, s in enumerate(content["conversations"]):
            if s["from"] != roles[j % 2]:
                valid = False
                break

        if valid:
            new_content.append(content)

    return new_content


def main(args):
    contents = json.load(open(args.in_file, "r"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='right',
        model_max_length=args.max_length,
        use_fast=False,
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,
    )
    print("Splitting long conversations...")
    new_content = split_all(contents, tokenizer, args.max_length)
    print(f"#in: {len(contents)}, #out: {len(new_content)}")
    print("Filtering invalid roles...")
    new_content = filter_invalid_roles(new_content)
    print(f"#in: {len(contents)}, #out: {len(new_content)}")

    with open(args.out_file, "w") as file:
        json.dump(new_content, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    main(args)
