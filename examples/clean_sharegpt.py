"""Prepare all datasets."""

import argparse
import json
import re
from typing import Any, Dict, List, Tuple


def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(in_file):
    with open(in_file, 'r') as f:
        json_data = json.load(f)
    return json_data


wrong_indices_pattern = re.compile('\n1\. [^2]*\n1\. ')


def should_skip(conv):
    # Filter wrong list indices like https://sharegpt.com/c/1pREAGO
    for sentence in conv['conversations']:
        val = sentence['value']
        sub = re.search(wrong_indices_pattern, val)
        if sub is not None:
            return True

    return False


def get_statistics(
        raw_data: List[Dict[str,
                            any]]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Get statistics from raw_data.

    Args:
        raw_data: A list of dictionaries containing conversation data.

    Returns:
        A tuple containing the role list and a dictionary of role occurrences per ID.
    """
    role_list = []
    role_res = {}

    for idx, raw_txt in enumerate(raw_data):
        id = raw_txt.get('id', str(idx))
        if idx % 10000 == 0:
            print(f'Processing {idx} / {len(raw_data)}')

        convs = raw_txt.get('conversations', [])
        role_res[id] = []

        for conv in convs:
            sender = conv.get('from')
            role_res[id].append(sender)

            if sender not in role_list:
                role_list.append(sender)

    return role_list, role_res


def format_roles(
    raw_data: List[Dict[str, List[Dict[str, str]]]]
) -> List[Dict[str, List[Dict[str, str]]]]:
    """Format the roles of conversations in raw_data.

    Args:
        raw_data: A list of dictionaries containing conversation data.

    Returns:
        A list of dictionaries containing formatted conversation data.
    """
    users = ['human', 'user']
    bots = ['gpt', 'bard', 'bing', 'chatgpt']
    role_list = users + bots
    collect_data = []

    for idx, raw_txt in enumerate(raw_data):
        convs = raw_txt.get('conversations', [])
        id = raw_txt.get('id', str(idx))
        new_convs = []

        for j, conv in enumerate(convs):
            sender = conv.get('from')

            if sender not in role_list:
                print(
                    f"Warning: Role '{sender}' is not recognized. Skipping conversation."
                )
                continue

            if sender in users[1:]:
                print(f"Correcting '{sender}' to '{users[0]}'")
                conv['from'] = users[0]

            if sender in bots[1:]:
                print(f"Correcting '{sender}' to '{bots[0]}'")
                conv['from'] = bots[0]

            if conv['from'] and conv['value']:
                new_convs.append(conv)

        if len(new_convs) >= 2:
            collect_data.append({'id': id, 'conversations': new_convs})
        else:
            print(f'Warning: Skipping conversation {idx}.', new_convs)
    return collect_data


def filter_invalid_roles(
        raw_data: List[Dict[str,
                            any]]) -> List[Dict[str, List[Dict[str, any]]]]:
    """
    Filter out invalid contents based on the roles assigned to each conversation.

    Args:
        raw_data: A list of dictionaries containing conversation data.

    Returns:
        A list of dictionaries containing filtered conversation data.
    """

    roles = ['human', 'gpt']
    filtered_data = []

    for idx, contents in enumerate(raw_data):
        # Get conversations and id from the current dictionary
        convs = contents.get('conversations', [])
        id = contents.get('id', str(idx))

        # Remove first conversation if it is not from 'human' role
        if convs and convs[0].get('from') != 'human':
            convs = convs[1:]

        # Check if number of conversations is less than 2
        if len(convs) < 2:
            continue

        # Truncate convs to have an even number of conversations
        convs = convs[:len(convs) // 2 * 2]

        valid = True
        for j, conv in enumerate(convs):
            # Check if role of conversation alternates between 'human' and 'gpt'
            if conv.get('from') != roles[j % 2]:
                valid = False
                break

        assert len(convs) % 2 == 0, 'Number of conversations must be even.'

        if valid:
            # Append filtered data to the result
            filtered_data.append({'id': id, 'conversations': convs})

    return filtered_data


def filter_wrong_format(raw_data):
    collect_data = []
    for raw_txt in raw_data:
        if should_skip(raw_txt):
            print(f"{raw_txt['id']} contains a wrong format.")
            print(raw_txt)
        else:
            collect_data.append(raw_txt)

    print(f'#in: {len(raw_data)}, #out: {len(collect_data)}')
    return collect_data


def get_clean_data(args: Any, save_stata_res: bool = False) -> Any:
    """Get clean data by processing raw data using helper functions.

    Args:
        args: Arguments passed to the function.

    Returns:
        Cleaned data after processing.
    """
    # Load raw data from file
    with open(args.in_file, 'r') as file:
        raw_data = json.load(file)

    # Get statistics for raw_data
    print('Getting statistics for raw_data...')
    res1, res2 = get_statistics(raw_data)

    if save_stata_res:
        # Save role_list and role_res to JSON files
        json_dump(res1, 'role_list.json')
        json_dump(res2, 'role_res.json')

    # Format roles in raw_data
    print('=' * 100)
    print('Formatting roles in raw_data...')
    clean_data1 = format_roles(raw_data)

    # Get statistics for clean_data1
    print('=' * 100)
    print('Getting statistics for clean_data1...')
    res1, res2 = get_statistics(clean_data1)

    if save_stata_res:
        # Save role_list_1 and role_res_1 to JSON files
        json_dump(res1, 'role_list_clean.json')
        json_dump(res2, 'role_res_clean.json')

    # Filter out incorrect data from clean_data1
    print('=' * 100)
    print('Filtering out incorrect data from clean_data1...')
    clean_data2 = filter_invalid_roles(clean_data1)
    # Print lengths of raw data, clean data1, and clean data2
    print(f'raw data len: {len(raw_data)}')
    print(f'clean data1 len: {len(clean_data1)}')
    print(f'clean data2 len: {len(clean_data2)}')
    return clean_data2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str)
    parser.add_argument('--out-file', type=str)
    args = parser.parse_args()
    clean_data2 = get_clean_data(args)
    json_dump(clean_data2, args.out_file)
