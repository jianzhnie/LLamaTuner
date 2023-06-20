import os
from typing import Dict, Optional


def get_dataset_path(dataset_name: str,
                     data_dir: Optional[str] = None,
                     load_from_local: bool = False) -> str:
    """
    Returns the path to a dataset based on a pre-defined map of dataset names to their corresponding URLs on the internet
    or local file paths.

    Args:
        dataset_name (str): The name of the dataset to load.
        data_dir (Optional[str]): The local directory where the dataset is stored; required only if load_from_local is True.
        load_from_local (bool): Whether to load the dataset from the internet (False) or from a local directory (True).

    Raises:
        KeyError: If dataset_name does not exist in the dataset_map.

    Returns:
        str: The path to the specified dataset.
    """
    if not load_from_local:
        dataset_map: Dict[str, str] = {
            'alpaca': 'tatsu-lab/alpaca',
            'alpaca-clean': 'yahma/alpaca-cleaned',
            'chip2': 'laion/OIG',
            'self-instruct': 'yizhongw/self_instruct',
            'hh-rlhf': 'Anthropic/hh-rlhf',
            'longform': 'akoksal/LongForm',
            'oasst1': 'timdettmers/openassistant-guanaco',
            'vicuna': 'anon8231489123/ShareGPT_Vicuna_unfiltered',
            'evol_instruct': 'WizardLM/WizardLM_evol_instruct_V2_196k',
            'dolly-15k': 'databricks/databricks-dolly-15k',
        }
    else:
        assert data_dir is not None
        dataset_map = {
            'alpaca':
            'tatsu-lab/alpaca',
            'alpaca-clean':
            'yahma/alpaca-cleaned',
            'chip2':
            'laion/OIG',
            'self-instruct':
            'yizhongw/self_instruct',
            'hh-rlhf':
            'Anthropic/hh-rlhf',
            'longform':
            'akoksal/LongForm',
            'evol_instruct':
            os.path.join(
                data_dir,
                'WizardLM/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.json'
            ),
            'oasst1':
            os.path.join(
                data_dir,
                'timdettmers/openassistant-guanaco/openassistant_best_replies_train.jsonl'
            ),
            'vicuna':
            os.path.join(
                data_dir,
                'anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json'
            ),
            'dolly-15k':
            'databricks/databricks-dolly-15k',
        }
    return dataset_map[dataset_name]
