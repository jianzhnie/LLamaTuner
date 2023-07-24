import json

from datasets import load_dataset


def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(in_file):
    with open(in_file, 'r') as f:
        json_data = json.load(f)
    return json_data


def convert_100PoisonMpts(in_file, out_file):
    clean_data = load_dataset('json', data_files=in_file)['train']
    new_content = []
    for i, raw_text in enumerate(clean_data):
        q = raw_text['prompt']
        a = raw_text['answer']
        new_content.append({
            'instruction': q,
            'input': '',
            'output': a,
        })

    print(f'#out: {len(new_content)}')
    json_dump(new_content, out_file)


if __name__ == '__main__':
    data_path = '/home/robin/prompt_data/100PoisonMpts/train.jsonl'
    out_path = '/home/robin/prompt_data/100PoisonMpts/train_alpaca.json'
    clean_data = load_dataset('json', data_files=data_path)
    convert_100PoisonMpts(data_path, out_file=out_path)
