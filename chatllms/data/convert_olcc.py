"""
Convert alpaca dataset into sharegpt format.

Usage: python3 -m fastchat.data.convert_alpaca --in alpaca_data.json
"""

import argparse
import json


def convert_olcc(in_file, out_file):
    with open(in_file, 'r') as file:
        content = json.load(file)
        new_content = []
        for i, turns in enumerate(content):
            turn = turns['turns']
            q = turn[0]['text']
            a = turn[1]['text']
            new_content.append({
                'id':
                f'olcc_{i}',
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

    print(f'#out: {len(new_content)}')
    json.dump(new_content, open(out_file, 'w'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str)
    parser.add_argument('--out-file', type=str)
    args = parser.parse_args()
    convert_olcc(args.in_file, args.out_file)
