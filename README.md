<div align="center">
  <img src="assets/guanaco.svg" width="300"/>
<div>&nbsp;</div>
</div>

![GitHub Repo stars](https://img.shields.io/github/stars/jianzhnie/Chinese-Guanaco?style=social)
![GitHub Code License](https://img.shields.io/github/license/jianzhnie/Chinese-Guanaco)
![GitHub last commit](https://img.shields.io/github/last-commit/jianzhnie/Chinese-Guanaco)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)
![issue resolution](https://img.shields.io/github/issues-closed-raw/jianzhnie/LLamaTuner)
![open issues](https://img.shields.io/github/issues-raw/jianzhnie/LLamaTuner)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<div align="center">

👋🤗🤗👋 Join our [WeChat](assets/wechat.jpg).

</div>

# Easy and Efficient Fine-tuning LLMs  --- 简单高效的大语言模型训练/部署

<div align="center">

[中文](README_zh.md) | English

</div>

## Introduction

LLamaTuner is an efficient, flexible and full-featured toolkit for fine-tuning LLM (Llama3, Phi3, Qwen, Mistral, ...)

**Efficient**

- Support LLM, VLM pre-training / fine-tuning on almost all GPUs. LLamaTuner is capable of fine-tuning 7B LLM on a single 8GB GPU, as well as multi-node fine-tuning of models exceeding 70B.
- Automatically dispatch high-performance operators such as FlashAttention and Triton kernels to increase training throughput.
- Compatible with [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀, easily utilizing a variety of ZeRO optimization techniques.

**Flexible**

- Support various LLMs ([Llama 3](https://huggingface.co/meta-llama), [Mixtral](https://huggingface.co/mistralai), [Llama 2](https://huggingface.co/meta-llama), [ChatGLM](https://huggingface.co/THUDM), [Qwen](https://huggingface.co/Qwen), [Baichuan](https://huggingface.co/baichuan-inc), ...).
- Support VLM ([LLaVA](https://github.com/haotian-liu/LLaVA)).
- Well-designed data pipeline, accommodating datasets in any format, including but not limited to open-source and custom formats.
- Support various training algorithms ([QLoRA](http://arxiv.org/abs/2305.14314), [LoRA](http://arxiv.org/abs/2106.09685), full-parameter fune-tune), allowing users to choose the most suitable solution for their requirements.

**Full-featured**

- Support continuous pre-training, instruction fine-tuning, and agent fine-tuning.
- Support chatting with large models with pre-defined templates.

## Table of Contents
- [Easy and Efficient Fine-tuning LLMs  --- 简单高效的大语言模型训练/部署](#easy-and-efficient-fine-tuning-llms------简单高效的大语言模型训练部署)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Supported Models](#supported-models)
  - [Supported Training Approaches](#supported-training-approaches)
  - [Supported Datasets](#supported-datasets)
    - [Data Preprocessing](#data-preprocessing)
  - [Model Zoo](#model-zoo)
  - [Requirement](#requirement)
    - [Hardware Requirement](#hardware-requirement)
  - [Getting Started](#getting-started)
    - [Clone the code](#clone-the-code)
  - [Getting Started](#getting-started-1)
    - [QLora int4 Finetune](#qlora-int4-finetune)
  - [Known Issues and Limitations](#known-issues-and-limitations)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Some lmm fine-tuning repos](#some-lmm-fine-tuning-repos)
  - [Citation](#citation)


## Supported Models

| Model                                                | Model size                       | Default module  | Template  |
| ---------------------------------------------------- | -------------------------------- | --------------- | --------- |
| [Baichuan](https://huggingface.co/baichuan-inc)      | 7B/13B                           | W_pack          | baichuan  |
| [Baichuan2](https://huggingface.co/baichuan-inc)     | 7B/13B                           | W_pack          | baichuan2 |
| [BLOOM](https://huggingface.co/bigscience)           | 560M/1.1B/1.7B/3B/7.1B/176B      | query_key_value | -         |
| [BLOOMZ](https://huggingface.co/bigscience)          | 560M/1.1B/1.7B/3B/7.1B/176B      | query_key_value | -         |
| [ChatGLM3](https://huggingface.co/THUDM)             | 6B                               | query_key_value | chatglm3  |
| [Command-R](https://huggingface.co/CohereForAI)      | 35B/104B                         | q_proj,v_proj   | cohere    |
| [DeepSeek (MoE)](https://huggingface.co/deepseek-ai) | 7B/16B/67B/236B                  | q_proj,v_proj   | deepseek  |
| [Falcon](https://huggingface.co/tiiuae)              | 7B/11B/40B/180B                  | query_key_value | falcon    |
| [Gemma/CodeGemma](https://huggingface.co/google)     | 2B/7B                            | q_proj,v_proj   | gemma     |
| [InternLM2](https://huggingface.co/internlm)         | 7B/20B                           | wqkv            | intern2   |
| [LLaMA](https://github.com/facebookresearch/llama)   | 7B/13B/33B/65B                   | q_proj,v_proj   | -         |
| [LLaMA-2](https://huggingface.co/meta-llama)         | 7B/13B/70B                       | q_proj,v_proj   | llama2    |
| [LLaMA-3](https://huggingface.co/meta-llama)         | 8B/70B                           | q_proj,v_proj   | llama3    |
| [LLaVA-1.5](https://huggingface.co/llava-hf)         | 7B/13B                           | q_proj,v_proj   | vicuna    |
| [Mistral/Mixtral](https://huggingface.co/mistralai)  | 7B/8x7B/8x22B                    | q_proj,v_proj   | mistral   |
| [OLMo](https://huggingface.co/allenai)               | 1B/7B                            | q_proj,v_proj   | -         |
| [PaliGemma](https://huggingface.co/google)           | 3B                               | q_proj,v_proj   | gemma     |
| [Phi-1.5/2](https://huggingface.co/microsoft)        | 1.3B/2.7B                        | q_proj,v_proj   | -         |
| [Phi-3](https://huggingface.co/microsoft)            | 3.8B                             | qkv_proj        | phi       |
| [Qwen](https://huggingface.co/Qwen)                  | 1.8B/7B/14B/72B                  | c_attn          | qwen      |
| [Qwen1.5 (Code/MoE)](https://huggingface.co/Qwen)    | 0.5B/1.8B/4B/7B/14B/32B/72B/110B | q_proj,v_proj   | qwen      |
| [StarCoder2](https://huggingface.co/bigcode)         | 3B/7B/15B                        | q_proj,v_proj   | -         |
| [XVERSE](https://huggingface.co/xverse)              | 7B/13B/65B                       | q_proj,v_proj   | xverse    |
| [Yi (1/1.5)](https://huggingface.co/01-ai)           | 6B/9B/34B                        | q_proj,v_proj   | yi        |
| [Yi-VL](https://huggingface.co/01-ai)                | 6B/34B                           | q_proj,v_proj   | yi_vl     |
| [Yuan](https://huggingface.co/IEITYuan)              | 2B/51B/102B                      | q_proj,v_proj   | yuan      |


## Supported Training Approaches

| Approach               | Full-tuning        | Freeze-tuning      | LoRA               | QLoRA              |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Pre-Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Supervised Fine-Tuning | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Reward Modeling        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| PPO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| DPO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| KTO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| ORPO Training          | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Supported Datasets

As of now, we support the following datasets, most of which are all available in the [Hugging Face datasets library](https://huggingface.co/datasets/).

<details><summary>Supervised fine-tuning dataset</summary>

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
- [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
- [OpenHermes](https://huggingface.co/datasets/teknium/openhermes)

</details>


<details><summary>Preference datasets</summary>

- [DPO mixed (en&zh)](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k)
- [Orca DPO Pairs (en)](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Open Assistant(en&zh)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
- [Orca DPO (de)](https://huggingface.co/datasets/mayflowergmbh/intel_orca_dpo_pairs_de)
- [KTO mixed (en)](https://huggingface.co/datasets/argilla/kto-mix-15k)
</details>


Please refer to [data/README.md](data/README.md) to learn how to use these datasets.  If you want to explore more datasets, please refer to the [awesome-instruction-datasets](https://github.com/jianzhnie/awesome-instruction-datasets). Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

### Data Preprocessing

We provide a number of data preprocessing tools in the [data](./llamatuner/data) folder. These tools are intended to be a starting point for further research and development.

- [data_utils.py](./llamatuner/data/data_utils.py) :  Data preprocessing and formatting
- [sft_dataset.py](./llamatuner/data/sft_dataset.py) :  Supervised fine-tuning dataset class and collator
- [conv_dataset.py](./llamatuner/data/conv_dataset.py) :  Conversation dataset class and collator

## Model Zoo

We provide a number of models in the [Hugging Face model hub](https://huggingface.co/decapoda-research). These models are trained with QLoRA and can be used for inference and finetuning. We provide the following models:

| Base Model                                                       | Adapter      | Instruct Datasets                                                                          | Train Script                                              | Log                                                               | Model on Huggingface                                                                |
| ---------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [llama-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | FullFinetune | -                                                                                          | -                                                         | -                                                                 |                                                                                     |
| [llama-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | QLoRA        | [openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) | [finetune_lamma7b](./scripts/finetune_llama_guanaco7b.sh) | [wandb log](https://wandb.ai/jianzhnie/huggingface/runs/1e2km7b1) | [GaussianTech/llama-7b-sft](https://huggingface.co/GaussianTech/llama-7b-sft)       |
| [llama-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | QLoRA        | [OL-CC](https://data.baai.ac.cn/details/OL-CC)                                             | [finetune_lamma7b](./scripts/finetune_llama_guanaco7b.sh) |                                                                   |                                                                                     |
| [baichuan7b](https://huggingface.co/baichuan-inc/baichuan-7B)    | QLoRA        | [openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) | [finetune_baichuan7b](./scripts/finetune_baichuan_7b.sh)  | [wandb log](https://wandb.ai/jianzhnie/huggingface/runs/41lq9joa) | [GaussianTech/baichuan-7b-sft](https://huggingface.co/GaussianTech/baichuan-7b-sft) |
| [baichuan7b](https://huggingface.co/baichuan-inc/baichuan-7B)    | QLoRA        | [OL-CC](https://data.baai.ac.cn/details/OL-CC)                                             | [finetune_baichuan7b](./scripts/finetune_baichuan_7b.sh)  | [wandb log](https://wandb.ai/jianzhnie/huggingface/runs/1lw2bmvn) | -                                                                                   |

## Requirement

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.8     | 3.10      |
| torch        | 1.13.1  | 2.2.0     |
| transformers | 4.37.2  | 4.41.0    |
| datasets     | 2.14.3  | 2.19.1    |
| accelerate   | 0.27.2  | 0.30.1    |
| peft         | 0.9.0   | 0.11.1    |
| trl          | 0.8.2   | 0.8.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.14.0    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.0   | 0.4.2     |
| flash-attn   | 2.3.0   | 2.5.8     |

### Hardware Requirement

\* *estimated*

| Method            | Bits | 7B    | 13B   | 30B   | 70B    | 110B   | 8x7B  | 8x22B  |
| ----------------- | ---- | ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| Full              | AMP  | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| Full              | 16   | 60GB  | 120GB | 300GB | 600GB  | 900GB  | 400GB | 1200GB |
| Freeze            | 16   | 20GB  | 40GB  | 80GB  | 200GB  | 360GB  | 160GB | 400GB  |
| LoRA/GaLore/BAdam | 16   | 16GB  | 32GB  | 64GB  | 160GB  | 240GB  | 120GB | 320GB  |
| QLoRA             | 8    | 10GB  | 20GB  | 40GB  | 80GB   | 140GB  | 60GB  | 160GB  |
| QLoRA             | 4    | 6GB   | 12GB  | 24GB  | 48GB   | 72GB   | 30GB  | 96GB   |
| QLoRA             | 2    | 4GB   | 8GB   | 16GB  | 24GB   | 48GB   | 18GB  | 48GB   |


## Getting Started

### Clone the code

Clone this repository and navigate to the Efficient-Tuning-LLMs folder

```bash
git clone https://github.com/jianzhnie/LLamaTuner.git
cd LLamaTuner
```

## Getting Started

| main function                    | Useage                                                                               | Scripts                                    |
| -------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------ |
| [train.py](tools/train.py)           | Full finetune LLMs on  SFT datasets                                                  | [full_finetune](./scripts/full_finetune)   |
| [train_lora.py](tools/train_lora.py) | Finetune LLMs by using Lora  (Low-Rank Adaptation of Large Language Models finetune) | [lora_finetune](./scripts/lora_finetune)   |
| [train_qlora.py](tools/rain_qlora.py) | Finetune LLMs by using QLora (QLoRA: Efficient Finetuning of Quantized LLMs)         | [qlora_finetune](./scripts/qlora_finetune) |

### QLora int4 Finetune

The `train_qlora.py` code is a starting point for finetuning and inference on various datasets.
Basic command for finetuning a baseline model on the Alpaca dataset:

```bash
python train_qlora.py --model_name_or_path <path_or_name>
```

For models larger than 13B, we recommend adjusting the learning rate:

```bash
python train_qlora.py –learning_rate 0.0001 --model_name_or_path <path_or_name>
```
To find more scripts for finetuning and inference, please refer to the `scripts` folder.



## Known Issues and Limitations

Here a list of known issues and bugs. If your issue is not reported here, please open a new issue and describe the problem.

1. 4-bit inference is slow. Currently, our 4-bit inference implementation is not yet integrated with the 4-bit matrix multiplication
2. Resuming a LoRA training run with the Trainer currently runs on an error
3. Currently, using `bnb_4bit_compute_type='fp16'` can lead to instabilities. For 7B LLaMA, only 80% of finetuning runs complete without error. We have solutions, but they are not integrated yet into bitsandbytes.
4. Make sure that `tokenizer.bos_token_id = 1` to avoid generation issues.

## License

`LLamaTuner` is released under the Apache 2.0 license.

## Acknowledgements

We thank the Huggingface team, in particular Younes Belkada, for their support integrating QLoRA with PEFT and transformers libraries.

We appreciate the work by many open-source contributors, especially:


- [LLaMa](https://github.com/facebookresearch/llama/)
- [Vicuna](https://github.com/lm-sys/FastChat/)
- [xTuring](https://github.com/stochasticai/xTuring)
- [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Hugging Face](https://huggingface.co/)
- [Peft](https://github.com/huggingface/peft.git)
- [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [deepspeed](https://www.deepspeed.ai/)
- [Unsloth](https://github.com/unslothai/unsloth)
- [qlora](https://github.com/artidoro/qlora)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)


## Some lmm fine-tuning repos

- https://github.com/QwenLM
- https://github.com/InternLM
- https://github.com/ymcui/Chinese-LLaMA-Alpaca-3
- https://github.com/ymcui/Chinese-Mixtral
- https://github.com/SmartFlowAI/EmoLLM
- https://github.com/yangjianxin1/Firefly
- https://github.com/LiuHC0428/LAW-GPT

## Citation

Please cite the repo if you use the data or code in this repo.

```bibtex
@misc{Chinese-Guanaco,
  author = {jianzhnie},
  title = {LLamaTuner: Easy and Efficient Fine-tuning LLMs},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jianzhnie/LLamaTuner}},
}
```
