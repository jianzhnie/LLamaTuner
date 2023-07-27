
## How to use the data

### Datasets Supported by the Framework

We provide the following datasets for the experiments in this framework.

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Stanford Alpaca (Chinese)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
- [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
- [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
- [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [mosaicml/dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf)
- [GPT-4 Generated Data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [Alpaca CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
- [UltraChat](https://github.com/thunlp/UltraChat)
- [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- [BIAI/OL-CC](https://data.baai.ac.cn/details/OL-CC)
- [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
- [Evol-Instruct](https://huggingface.co/datasets/victor123/evol_instruct_70k)

### Dataset formation

The `dataset_info.yaml` file contains the information of the datasets. The format of the file is as follows.

```yaml
dataset_name:
    hf_hub_url: # "the name of the dataset repository on the HuggingFace hub. (if specified, ignore below 3 arguments)",
    local_path: # "the name of the dataset file in the this directory. (required if above are not specified)",
    dataset_format: # "the format of the dataset. (required), e.g., alpaca, dolly, etc.",
    multi_turn:  # "whether the dataset is multi-turn. (default: False)"
```

For example, the following is the dataset information of the Stanford Alpaca dataset.

```yaml
alpaca:
  hf_hub_url: tatsu-lab/alpaca
  local_path:
  dataset_format: alpaca
  multi_turn: False
```
This will load the dataset from the HuggingFace hub. If you want to load the dataset from local files, please specify the `local_path` field.

```yaml
alpaca:
  hf_hub_url: tatsu-lab/alpaca
  local_path: path/to/alpaca.json
  dataset_format: alpaca
  multi_turn: False
```

### Custom datasets

If you are using a custom dataset, please provide your dataset definition in  `dataset_info.yaml`.

#### hf_hub_url and local_path

By defaullt, the framework will load the datasets from the HuggingFace hub. If you want to use the datasets from local files, please specify the `local_path` in the `dataset_info.yaml` file.

#### dataset_format
As for the dataset_format field, which is used to specify the format of the dataset, will good for the framework to process the dataset. Currently, we support the following dataset formats.

- `alpaca`: Alpaca dataset
- `dolly`: Dolly dataset
- `gpt4`: GPT-4 generated dataset
- `alpaca_cot`: Alpaca CoT dataset
- `oasst1`: OpenAssistant/oasst1 dataset
- `sharegpt`: Multi-turn ShareGPT dataset

If your dataset is not in the above format, there are two ways to use it.

- The first way, Implement the `format_dataset` function in `./chatllms/data/data_utils.py` to formate your dataset. For example, the following is the `_format_dolly15k` function for the Dolly dataset.

```python
def _format_dolly15k(dataset: Dataset) -> Dataset:
    """Format Dolly-15k dataset."""
    dataset = dataset.rename_column('context', 'input')
    dataset = dataset.rename_column('response', 'output')
    return dataset
```

- The second way, convert your dataset to the above format and specify the `dataset_format` field in `dataset_info.yaml`.

For example, if we want to convert the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) into  Alpaca formate, you can refer to the following code.

```python
import json
def convert_dolly_alpaca(in_file, out_file):
    with open(in_file, 'r') as file:
        contents = json.load(file)
        new_content = []
        for i, content in enumerate(contents):
            new_content.append({
              'instruction': content['instruction'],
              'input': content['text'],
              'output': content['text'],
            })

    print(f'#out: {len(new_content)}')
    with open(out_file, 'w') as file:
        json.dump(new_content, file, indent=2, ensure_ascii=False)
```

#### multi_turn

If your dataset is multi-turn, please specify the `multi_turn` field in `dataset_info.yaml`. The framework will automatically process the multi-turn dataset. Flowing is an example of the multi-turn dataset.

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
      },
      {
        "from": "human",
        "value": "What can you do?"
      },
      {
        "from": "gpt",
        "value": "I can chat with you."
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "My name is Vicuna, and I'm a language model developed by Large Model Systems Organization (LMSYS)."
      }
    ]
  },
]
```

For now, we only support the multi-turn dataset in the above format. If your dataset is not in the above format, please convert it. We also provide the following code to convert the Dolly dataset to the above format. You can find the code in `./chatllms/data/utils/convert_alpaca.py`.

```python
import argparse
import json
from typing import Any, Dict, List

from datasets import load_dataset

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
    new_data = convert_dolly_vicuna(raw_data)
    json_dump(new_data, args.out_file)


if __name__ == '__main__':
    main()
```
