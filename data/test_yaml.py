import yaml

# 读取YAML文件
with open('dataset_info.yaml') as f:
    config = yaml.safe_load(f)

print(config)

dataset_dict = {}

for name, info in config.items():
    dataset_dict[name] = {
        'hf_hub_url': info['hf_hub_url'],
        'local_path': info['local_path'],
        'multi_turn': info['multi_turn'],
    }

    if 'columns' in info:
        dataset_dict[name]['columns'] = info['columns']

print(dataset_dict)
