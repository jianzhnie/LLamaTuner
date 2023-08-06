import json
import sys

from datasets import load_dataset

sys.path.append('../../')

from chatllms.data.data_utils import extract_default_prompt_dataset


def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(in_file):
    with open(in_file, 'r') as f:
        json_data = json.load(f)
    return json_data


def valid_keys(keys):
    for k in ['input', 'output']:
        if k not in keys:
            return False
    return True


def remove_unused_columns(dataset):
    """Remove columns not named 'input' or 'output'."""
    dataset = dataset.remove_columns([
        col for col in dataset.column_names if col not in ['input', 'output']
    ])
    return dataset


def convert_alpaca_vicuna(in_file: str, out_file: str = None):
    raw_dataset = load_dataset('json', data_files=in_file)['train']
    raw_dataset = raw_dataset.map(extract_default_prompt_dataset)

    collect_data = []
    for i, content in enumerate(raw_dataset):
        prompt = content['input']
        response = content['output']

        collect_data.append({
            'id':
            f'alpaca_{i}',
            'conversations': [
                {
                    'from': 'human',
                    'value': prompt
                },
                {
                    'from': 'gpt',
                    'value': response
                },
            ],
        })
    print(f'Original: {len(raw_dataset)}, Converted: {len(collect_data)}')
    json_dump(collect_data, out_file)
    return collect_data


if __name__ == '__main__':
    in_file = '/home/robin/prompt_data/100PoisonMpts/train_alpaca.json'
    out_file = '/home/robin/prompt_data/100PoisonMpts/train_vicuna.json'
    collect_data = convert_alpaca_vicuna(in_file, out_file)

    data_path = '/home/robin/prompt_data/CValues-Comparison/test_alpaca.json'
    out_path = '/home/robin/prompt_data/CValues-Comparison/test_vicuna.json'
    convert_alpaca_vicuna(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/CValues-Comparison/train_alpaca.json'
    out_path = '/home/robin/prompt_data/CValues-Comparison/train_vicuna.json'
    convert_alpaca_vicuna(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/HuatuoGPT-sft-data-v1/HuatuoGPT_alpaca.json'
    out_path = '/home/robin/prompt_data/HuatuoGPT-sft-data-v1/HuatuoGPT_vicnua.json'
    convert_alpaca_vicuna(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/Safety-Prompts/attack_scenarios_alpaca.json'
    out_path = '/home/robin/prompt_data/Safety-Prompts/attack_scenarios_vicuna.json'
    convert_alpaca_vicuna(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/Safety-Prompts/safety_scenarios_alpaca.json'
    out_path = '/home/robin/prompt_data/Safety-Prompts/safety_scenarios_vicuna.json'
    convert_alpaca_vicuna(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/COIG/train_alpaca.json'
    out_path = '/home/robin/prompt_data/COIG/train_vicuna.json'
    convert_alpaca_vicuna(data_path, out_file=out_path)
