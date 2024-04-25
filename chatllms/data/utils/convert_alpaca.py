"""Convert alpaca dataset into sharegpt format.

Usage: python3 -m chatllms.data.convert_alpaca --in alpaca_data.json
"""

import argparse
import json
from typing import Any, Dict, List

from datasets import load_dataset


def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(in_file):
    with open(in_file, 'r') as f:
        json_data = json.load(f)
    return json_data


def valid_keys(keys):
    for k in ['instruction', 'input', 'output']:
        if k not in keys:
            return False
    return True


def convert_alpaca_vicuna(raw_data: List[Dict[str, Any]]):
    collect_data = []
    for i, content in enumerate(raw_data):
        if not valid_keys(content.keys()):
            continue

        if len(content['input'].strip()) > 1:
            q, a = content['instruction'] + '\nInput:\n' + content[
                'input'], content['output']
        else:
            q, a = content['instruction'], content['output']

        collect_data.append({
            'id':
            f'alpaca_{i}',
            'conversations': [
                {
                    'from': 'human',
                    'value': q
                },
                {
                    'from': 'gpt',
                    'value': a
                },
            ],
        })
    print(f'Original: {len(raw_data)}, Converted: {len(collect_data)}')
    return collect_data


def convert_dolly_vicuna(raw_data: List[Dict[str, Any]]):
    collect_data = []
    for i, content in enumerate(raw_data):
        if len(content['context'].strip()) > 1:
            q, a = content['instruction'] + '\nInput:\n' + content[
                'context'], content['response']
        else:
            q, a = content['instruction'], content['response']

        collect_data.append({
            'id':
            f'alpaca_{i}',
            'conversations': [
                {
                    'from': 'human',
                    'value': q
                },
                {
                    'from': 'gpt',
                    'value': a
                },
            ],
        })
    print(f'Original: {len(raw_data)}, Converted: {len(collect_data)}')
    return collect_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str)
    parser.add_argument('--out-file', type=str)
    args = parser.parse_args()

    raw_data = load_dataset('json', data_files=args.in_file)['train']
    new_data = convert_alpaca_vicuna(raw_data)

    # new_data = convert_dolly_vicuna(raw_data)
    # new_data = convert_alpaca_vicuna(raw_data)
    json_dump(new_data, args.out_file)


if __name__ == '__main__':
    main()
