import argparse
import json
import os
import random


def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(in_file):
    with open(in_file, 'r') as f:
        json_data = json.load(f)
    return json_data


def convert_oasst1_data(data_dir, output_dir):
    """For OASST1, because it's in a tree structure, where every user input
    might get multiple replies, we have to save every path from the root node
    to the assistant reply (including both leaf node and intemediate node).

    This results in some of the messages being duplicated among different paths
    (instances). Be careful when using this dataset for training. Ideally, you
    should only minimize the loss of the last message in each path.
    """
    conversations = []
    with open(os.path.join(data_dir, '2023-04-12_oasst_ready.trees.jsonl'),
              'r') as fin:
        for line in fin:
            conversations.append(json.loads(line))

    output_path = os.path.join(output_dir, 'oasst1_data.jsonl')

    # tranvers the conversation tree, and collect all valid sequences
    def dfs(reply, messages, valid_sequences):
        if reply['role'] == 'assistant':
            messages.append({'role': 'assistant', 'content': reply['text']})
            valid_sequences.append(messages[:])
            for child in reply['replies']:
                dfs(child, messages, valid_sequences)
            messages.pop()
        elif reply['role'] == 'prompter':
            messages.append({'role': 'user', 'content': reply['text']})
            for child in reply['replies']:
                dfs(child, messages, valid_sequences)
            messages.pop()
        else:
            raise ValueError(f"Unknown role: {reply['role']}")

    with open(output_path, 'w') as fout:
        example_cnt = 0
        for _, conversation in enumerate(conversations):
            valid_sequences = []
            dfs(conversation['prompt'], [], valid_sequences)
            for sequence in valid_sequences:
                fout.write(
                    json.dumps({
                        'dataset': 'oasst1',
                        'id': f'oasst1_{example_cnt}',
                        'messages': sequence
                    }) + '\n')
                example_cnt += 1


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--raw_data_dir',
                            type=str,
                            default='data/downloads')
    arg_parser.add_argument('--output_dir', type=str, default='data/processed')
    arg_parser.add_argument('--seed', type=int, default=42)
    args = arg_parser.parse_args()
    random.seed(args.seed)

    convert_oasst1_data(data_dir=args.raw_data_dir, output_dir=args.output_dir)
