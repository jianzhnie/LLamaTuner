from os.path import join


def get_dataset_info(dataset_dir):
    """
    Returns the datasets info to a dataset based on a pre-defined map of dataset names to their corresponding URLs on the internet
    or local file paths.

    Args:
        dataset_dir (str): The local directory where the dataset is stored; this is used for datasets that are stored locally.

    Returns:
        str: The dataset dict to the specified dataset.
    """
    dataset_info = {
        'alpaca': {
            'hf_hub_url': 'tatsu-lab/alpaca',
            'local_path': 'tatsu-lab/alpaca/alpaca.json',
            'multi_turn': False
        },
        'alpaca-clean': {
            'hf_hub_url': 'yahma/alpaca-cleaned',
            'local_path': '',
            'multi_turn': False
        },
        'chip2': {
            'hf_hub_url': 'laion/OIG',
            'local_path': '',
            'multi_turn': False
        },
        'self-instruct': {
            'hf_hub_url': 'yizhongw/self_instruct',
            'local_path': '',
            'multi_turn': False
        },
        'guanaco': {
            'hf_hub_url': 'JosephusCheung/GuanacoDataset',
            'local_path': '',
            'multi_turn': False
        },
        'hh-rlhf': {
            'hf_hub_url': 'Anthropic/hh-rlhf',
            'local_path': '',
            'multi_turn': False
        },
        'longformer': {
            'hf_hub_url': 'akoksal/LongForm',
            'local_path': '',
            'multi_turn': False
        },
        'openassistant-guanaco': {
            'hf_hub_url':
            'timdettmers/openassistant-guanaco',
            'local_path':
            join(dataset_dir,
                 'timdettmers/openassistant_best_replies_train.jsonl'),
            'multi_turn':
            False
        },
        'evol_instruct': {
            'hf_hub_url':
            'WizardLM/WizardLM_evol_instruct_V2_196k',
            'local_path':
            join(dataset_dir, 'WizardLM/WizardLM_evol_instruct_V2_143k.json'),
            'multi_turn':
            False
        },
        'dolly-15k': {
            'hf_hub_url': 'databricks/databricks-dolly-15k',
            'local_path': join(dataset_dir, 'databricks/databricks-dolly-15k'),
            'multi_turn': False
        },
        'olcc': {
            'hf_hub_url': 'yizhongw/olcc',
            'local_path': join(dataset_dir, 'olcc/olcc_alpaca.json'),
            'multi_turn': False
        },
        'share_gpt': {
            'hf_hub_url': '',
            'local_path': join(dataset_dir, 'sharegpt/sharegpt_split.json'),
            'multi_turn': True
        },
        '100PoisonMpts': {
            'hf_hub_url': '',
            'local_path': join(dataset_dir, '100PoisonMpts/train.jsonl'),
            'multi_turn': False
        },
        'belle_0.5m': {
            'hf_hub_url': 'BelleGroup/train_0.5M_CN',
            'local_path': '',
            'multi_turn': False
        },
        'belle_1m': {
            'hf_hub_url': 'BelleGroup/train_1M_CN',
            'local_path': '',
            'multi_turn': False
        },
        'belle_2m': {
            'hf_hub_url': 'BelleGroup/train_2M_CN',
            'local_path': '',
            'multi_turn': False
        },
        'belle_dialog': {
            'hf_hub_url': 'BelleGroup/generated_chat_0.4M',
            'local_path': '',
            'multi_turn': False
        },
        'belle_math': {
            'hf_hub_url': 'BelleGroup/school_math_0.25M',
            'local_path': '',
            'multi_turn': False
        },
        'belle_multiturn': {
            'hf_hub_url': 'BelleGroup/multi_turn_0.5M',
            'local_path': '',
            'multi_turn': True,
            'columns': {
                'prompt': 'instruction',
                'query': '',
                'response': 'output',
                'history': 'history'
            }
        },
        'firefly': {
            'hf_hub_url': 'YeungNLP/firefly-train-1.1M',
            'local_path': '',
            'multi_turn': False,
            'columns': {
                'prompt': 'input',
                'query': '',
                'response': 'target',
                'history': ''
            }
        },
        'codealpaca': {
            'hf_hub_url': 'sahil2801/CodeAlpaca-20k',
            'local_path': '',
            'multi_turn': False
        },
        'alpaca_cot': {
            'hf_hub_url': 'QingyiSi/Alpaca-CoT',
            'local_path': '',
            'multi_turn': False
        },
        'webqa': {
            'hf_hub_url': 'suolyer/webqa',
            'local_path': '',
            'multi_turn': False,
            'columns': {
                'prompt': 'input',
                'query': '',
                'response': 'output',
                'history': ''
            }
        },
        'novel_tokens512_50k': {
            'hf_hub_url': 'zxbsmk/webnovel_cn',
            'local_path': '',
            'multi_turn': False
        }
    }

    return dataset_info
