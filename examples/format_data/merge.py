"""
Merge two conversation files into one

Usage: python3 -m fastchat.data.merge --in file1.json file2.json --out merged.json
"""

import argparse
import json


def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(in_file):
    with open(in_file, 'r') as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, required=True, nargs='+')
    parser.add_argument('--out-file', type=str, default='merged.json')
    args = parser.parse_args()

    new_content = []
    for in_file in args.in_file:
        content = json_load(in_file)
        print(f'in-file: {in_file}, len: {len(content)}')
        new_content.extend(content)

    print(f'#out: {len(new_content)}')
    print(f'Save new_content to {args.out_file}')
    json_dump(new_content, args.out_file)
