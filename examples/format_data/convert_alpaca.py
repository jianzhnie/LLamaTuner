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
    raw_data = load_dataset('json', data_files=in_file)['train']
    new_content = []
    for i, raw_text in enumerate(raw_data):
        prompt = raw_text['prompt']
        response = raw_text['answer']
        if len(prompt) <= 5 or len(response) <= 5:
            continue
        new_content.append({
            'instruction': prompt,
            'input': '',
            'output': response,
        })

    print(f'#out: {len(new_content)}')
    json_dump(new_content, out_file)


def convert_Cvalues(in_file, out_file):
    raw_data = load_dataset('json', data_files=in_file)['train']
    new_content = []
    for i, raw_text in enumerate(raw_data):
        prompt = raw_text['prompt']
        response = raw_text['pos_resp']
        if len(prompt) <= 5 or len(response) <= 5:
            continue
        new_content.append({
            'instruction': prompt,
            'input': '',
            'output': response,
        })

    print(f'#out: {len(new_content)}')
    json_dump(new_content, out_file)


def convert_huatuogpt(in_file, out_file):
    raw_data = load_dataset('json', data_files=in_file)['train']
    new_content = []
    for i, raw_text in enumerate(raw_data):
        data = raw_text['data']
        prompt = data[0].replace('问：', '')
        response = data[1].replace('答：', '')
        if len(prompt) <= 5 or len(response) <= 5:
            continue
        new_content.append({
            'instruction': prompt,
            'input': '',
            'output': response,
        })
    print(f'#out: {len(new_content)}')
    json_dump(new_content, out_file)


def convert_safety_attack(in_file, out_file):
    field_list = [
        'Reverse_Exposure', 'Goal_Hijacking', 'Prompt_Leaking',
        'Unsafe_Instruction_Topic', 'Role_Play_Instruction',
        'Inquiry_With_Unsafe_Opinion'
    ]
    new_content = []
    for filed in field_list:
        raw_data = load_dataset('json', field=filed,
                                data_files=in_file)['train']
        for i, raw_text in enumerate(raw_data):
            prompt = raw_text['prompt']
            response = raw_text['response']
            if len(prompt) <= 5 or len(response) <= 5:
                continue
            new_content.append({
                'instruction': prompt,
                'input': '',
                'output': response,
            })
    print(f'#out: {len(new_content)}')
    json_dump(new_content, out_file)


def convert_safety_scenarios(in_file, out_file):

    field_list = [
        'Unfairness_And_Discrimination', 'Crimes_And_Illegal_Activities',
        'Insult', 'Mental_Health', 'Physical_Harm', 'Privacy_And_Property',
        'Ethics_And_Morality'
    ]
    new_content = []
    for filed in field_list:
        raw_data = load_dataset('json', data_files=in_file,
                                field=filed)['train']
        for i, raw_text in enumerate(raw_data):
            prompt = raw_text['prompt']
            response = raw_text['response']
            if len(prompt) <= 5 or len(response) <= 5:
                continue
            new_content.append({
                'instruction': prompt,
                'input': '',
                'output': response,
            })
    print(f'#out: {len(new_content)}')
    json_dump(new_content, out_file)


if __name__ == '__main__':

    data_path = '/home/robin/prompt_data/100PoisonMpts/train.jsonl'
    out_path = '/home/robin/prompt_data/100PoisonMpts/train_alpaca.jsonl'
    convert_100PoisonMpts(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/CValues-Comparison/test.jsonl'
    out_path = '/home/robin/prompt_data/CValues-Comparison/test_alpaca.json'
    convert_Cvalues(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/CValues-Comparison/train.jsonl'
    out_path = '/home/robin/prompt_data/CValues-Comparison/train_alpaca.json'
    convert_Cvalues(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/HuatuoGPT-sft-data-v1/HuatuoGPT_sft_data_v1.jsonl'
    out_path = '/home/robin/prompt_data/HuatuoGPT-sft-data-v1/HuatuoGPT_alpaca.json'
    convert_huatuogpt(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/Safety-Prompts/instruction_attack_scenarios.json'
    out_path = '/home/robin/prompt_data/Safety-Prompts/attack_scenarios_alpaca.json'
    convert_safety_attack(data_path, out_file=out_path)

    data_path = '/home/robin/prompt_data/Safety-Prompts/typical_safety_scenarios.json'
    out_path = '/home/robin/prompt_data/Safety-Prompts/safety_scenarios_alpaca.json'
    convert_safety_scenarios(data_path, out_file=out_path)
