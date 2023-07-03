"""Prepare all datasets."""

import argparse
import json
from typing import Any, Dict, List, Tuple


def json_dump(json_data, out_file):
    with open(out_file, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def check_adjacent_duplicates(lst):
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
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
        id = raw_txt.get('id')
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
        id = raw_txt.get('id')
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

    return collect_data


def filter_wrong_data(
        raw_data: List[Dict[str,
                            any]]) -> List[Dict[str, List[Dict[str, any]]]]:
    """Filter out incorrect data from raw_data.

    Args:
        raw_data: A list of dictionaries containing conversation data.

    Returns:
        A list of dictionaries containing filtered conversation data.
    """
    roles = ['human', 'gpt']
    collect_data = []

    for idx, contents in enumerate(raw_data):
        convs = contents.get('conversations', [])
        id = contents.get('id')
        new_convs = []

        if convs[0].get('from') != 'human':
            convs = convs[1:]

        if len(convs) < 2:
            continue

        role_lst = [conv['from'] for conv in convs]

        if check_adjacent_duplicates(role_lst):
            continue
        else:
            for j, conv in enumerate(convs):
                if conv.get('from') == roles[j % 2]:
                    new_convs.append(conv)

            collect_data.append({'id': id, 'conversations': new_convs})

    return collect_data


def get_clean_data(args: Any) -> Any:
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

    # Save role_list_1 and role_res_1 to JSON files
    json_dump(res1, 'role_list_1.json')
    json_dump(res2, 'role_res_1.json')

    # Filter out incorrect data from clean_data1
    print('=' * 100)
    print('Filtering out incorrect data from clean_data1...')
    clean_data2 = filter_wrong_data(clean_data1)

    # Get statistics for clean_data2
    print('=' * 100)
    print('Getting statistics for clean_data2...')
    res1, res2 = get_statistics(clean_data2)

    # Save role_list_2 and role_res_2 to JSON files
    json_dump(res1, 'role_list_2.json')
    json_dump(res2, 'role_res_2.json')

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
    args.in_file = '/home/robin/prompt_data/anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json'
    args.out_file = '/home/robin/work_dir/llm/Chinese-Guanaco/examples/sharegpt_formate_role_filter.json'
    clean_data2 = get_clean_data(args)
    json_dump(clean_data2, args.out_file)
