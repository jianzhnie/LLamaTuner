
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

The `dataset_info.yaml` file contains the information of the datasets. By defaullt, the framework will load the datasets from the HuggingFace hub. If you want to use the datasets from local files, please specify the `local_path` in the `dataset_info.yaml` file. For example, if you want to use the Alpaca dataset from local files, please specify the following in `dataset_info.yaml`.

```yaml
alpaca:
  hf_hub_url: tatsu-lab/alpaca
  local_path: tatsu-lab/alpaca/alpaca.json
  multi_turn: False
```

### Custom datasets

If you are using a custom dataset, please provide your dataset definition in the following format in `dataset_info.yaml`.

```yaml
dataset_name:
    hf_hub_url: # "the name of the dataset repository on the HuggingFace hub. (if specified, ignore below 3 arguments)",
    local_path: # "the name of the dataset file in the this directory. (required if above are not specified)",
    multi_turn:  # "whether the dataset is multi-turn. (default: False)"
```

where the `prompt` and `response` columns should contain non-empty values. The `query` column will be concatenated with the `prompt` column and used as input for the model. The `history` column should contain a list where each element is a string tuple representing a query-response pair.
