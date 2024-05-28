<div align="center">
  <img src="assets/guanaco.svg" width="300"/>
<div>&nbsp;</div>
</div>

![GitHub Repo stars](https://img.shields.io/github/stars/jianzhnie/Chinese-Guanaco?style=social)
![GitHub Code License](https://img.shields.io/github/license/jianzhnie/Chinese-Guanaco)
![GitHub last commit](https://img.shields.io/github/last-commit/jianzhnie/Chinese-Guanaco)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![GitHub Tread](https://trendshift.io/api/badge/repositories/4535)](https://trendshift.io/repositories/4535)


<div align="center">

👋🤗🤗👋 Join our [WeChat](assets/wechat.jpg).

</div>

# Easy and Efficient Fine-tuning LLMs  --- 简单高效的大语言模型训练/部署

<div align="center">

[中文](README_zh.md) | English

</div>

## Table of Contents
- [Easy and Efficient Fine-tuning LLMs  --- 简单高效的大语言模型训练/部署](#easy-and-efficient-fine-tuning-llms------简单高效的大语言模型训练部署)
  - [Table of Contents](#table-of-contents)
  - [Supported Models](#supported-models)
  - [Supported Training Approaches](#supported-training-approaches)
  - [Supported Datasets](#supported-datasets)
    - [Data Preprocessing](#data-preprocessing)
  - [Model Zoo](#model-zoo)
  - [Installation](#installation)
    - [Requirement](#requirement)
    - [Install required packages](#install-required-packages)
    - [Clone the code](#clone-the-code)
  - [Getting Started](#getting-started)
    - [QLora int4 Finetune](#qlora-int4-finetune)
  - [Quantization](#quantization)
  - [Tutorials and Demonstrations](#tutorials-and-demonstrations)
  - [Using Local Datasets](#using-local-datasets)
  - [Multi GPU](#multi-gpu)
  - [Inference](#inference)
    - [终端交互式对话](#终端交互式对话)
    - [使用 Gradio 进行网页端交互](#使用-gradio-进行网页端交互)
  - [Sample Outputs](#sample-outputs)
  - [Known Issues and Limitations](#known-issues-and-limitations)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
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

We provide a number of data preprocessing tools in the [data](./chatllms/data) folder. These tools are intended to be a starting point for further research and development.

- [data_utils.py](./chatllms/data/data_utils.py) :  Data preprocessing and formatting
- [sft_dataset.py](./chatllms/data/sft_dataset.py) :  Supervised fine-tuning dataset class and collator
- [conv_dataset.py](./chatllms/data/conv_dataset.py) :  Conversation dataset class and collator

## Model Zoo

We provide a number of models in the [Hugging Face model hub](https://huggingface.co/decapoda-research). These models are trained with QLoRA and can be used for inference and finetuning. We provide the following models:

| Base Model                                                       | Adapter      | Instruct Datasets                                                                          | Train Script                                              | Log                                                               | Model on Huggingface                                                                |
| ---------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [llama-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | FullFinetune | -                                                                                          | -                                                         | -                                                                 |                                                                                     |
| [llama-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | QLoRA        | [openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) | [finetune_lamma7b](./scripts/finetune_llama_guanaco7b.sh) | [wandb log](https://wandb.ai/jianzhnie/huggingface/runs/1e2km7b1) | [GaussianTech/llama-7b-sft](https://huggingface.co/GaussianTech/llama-7b-sft)       |
| [llama-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | QLoRA        | [OL-CC](https://data.baai.ac.cn/details/OL-CC)                                             | [finetune_lamma7b](./scripts/finetune_llama_guanaco7b.sh) |                                                                   |                                                                                     |
| [baichuan7b](https://huggingface.co/baichuan-inc/baichuan-7B)    | QLoRA        | [openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) | [finetune_baichuan7b](./scripts/finetune_baichuan_7b.sh)  | [wandb log](https://wandb.ai/jianzhnie/huggingface/runs/41lq9joa) | [GaussianTech/baichuan-7b-sft](https://huggingface.co/GaussianTech/baichuan-7b-sft) |
| [baichuan7b](https://huggingface.co/baichuan-inc/baichuan-7B)    | QLoRA        | [OL-CC](https://data.baai.ac.cn/details/OL-CC)                                             | [finetune_baichuan7b](./scripts/finetune_baichuan_7b.sh)  | [wandb log](https://wandb.ai/jianzhnie/huggingface/runs/1lw2bmvn) | -                                                                                   |

## Installation

### Requirement

- CUDA >= 11.0

- Python 3.8+ and PyTorch 1.13.1+

- 🤗Transformers, Datasets, Accelerate, PEFT and bitsandbytes

- jieba, rouge_chinese and nltk (used at evaluation)

- gradio (used in gradio_webserver.py)

### Install required packages

To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source and make sure you have the latest version of the bitsandbytes library (0.39.0). You can achieve the above with the following commands:

```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

### Clone the code

Clone this repository and navigate to the Efficient-Tuning-LLMs folder

```bash
git clone https://github.com/jianzhnie/Efficient-Tuning-LLMs.git
cd Efficient-Tuning-LLMs
```

## Getting Started

| main function                    | Useage                                                                               | Scripts                                    |
| -------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------ |
| [train.py](./train.py)           | Full finetune LLMs on  SFT datasets                                                  | [full_finetune](./scripts/full_finetune)   |
| [train_lora.py](./train_lora.py) | Finetune LLMs by using Lora  (Low-Rank Adaptation of Large Language Models finetune) | [lora_finetune](./scripts/lora_finetune)   |
| [train_qlora.py](train_qlora.py) | Finetune LLMs by using QLora (QLoRA: Efficient Finetuning of Quantized LLMs)         | [qlora_finetune](./scripts/qlora_finetune) |

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

We can also tweak our hyperparameters:

```bash
python train_qlora.py \
    --model_name_or_path ~/checkpoints/baichuan7b \
    --dataset_cfg ./data/alpaca_zh_pcyn.yaml \
    --output_dir ./work_dir/oasst1-baichuan-7b \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_total_limit 5 \
    --save_steps 100 \
    --logging_strategy steps \
    --logging_steps 1 \
    --learning_rate 0.0002 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --lr_scheduler_type constant \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --max_new_tokens 32 \
    --source_max_len 512 \
    --target_max_len 512 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --double_quant \
    --quant_type nf4 \
    --fp16 \
    --bits 4 \
    --gradient_checkpointing \
    --trust_remote_code \
    --do_train \
    --do_eval \
    --sample_generate \
    --data_seed 42 \
    --seed 0
```

To find more scripts for finetuning and inference, please refer to the `scripts` folder.

## Quantization

Quantization parameters are controlled from the `BitsandbytesConfig` ([see HF documenation](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)) as follows:

- Loading in 4 bits is activated through `load_in_4bit`
- The datatype used for the linear layer computations with `bnb_4bit_compute_dtype`
- Nested quantization is activated through `bnb_4bit_use_double_quant`
- The datatype used for qunatization is specified with `bnb_4bit_quant_type`. Note that there are two supported quantization datatypes `fp4` (four bit float) and `nf4` (normal four bit float). The latter is theoretically optimal for normally distributed weights and we recommend using `nf4`.

```python
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path='/name/or/path/to/your/model',
        load_in_4bit=True,
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
```

## Tutorials and Demonstrations

We provide two Google Colab notebooks to demonstrate the use of 4bit models in inference and fine-tuning. These notebooks are intended to be a starting point for further research and development.

- [Basic usage Google Colab notebook](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing) - This notebook shows how to use 4bit models in inference with all their variants, and how to run GPT-neo-X (a 20B parameter model) on a free Google Colab instance 🤯
- [Fine tuning Google Colab notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) - This notebook shows how to fine-tune a 4bit model on a downstream task using the Hugging Face ecosystem. We show that it is possible to fine tune GPT-neo-X 20B on a Google Colab instance!

Other examples are found under the examples/ folder.

- Finetune LLama-7B (ex1)
- Finetune GPT-neo-X 20B (ex2)

## Using Local Datasets

You can specify the path to your dataset using the --dataset argument. If the --dataset_format argument is not set, it will default to the Alpaca format. Here are a few examples:

- Training with an alpaca format dataset:

```python
python train_qlora.py --dataset="path/to/your/dataset"
```

- Training with a self-instruct format dataset:

```python
python train_qlora.py --dataset="path/to/your/dataset" --dataset_format="self-instruct"
```

## Multi GPU

Multi GPU training and inference work out-of-the-box with Hugging Face's Accelerate. Note that the per_device_train_batch_size and per_device_eval_batch_size arguments are global batch sizes unlike what their name suggest.

When loading a model for training or inference on multiple GPUs you should pass something like the following to AutoModelForCausalLM.from_pretrained():

```python
device_map = "auto"
max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
```

## Inference

### 终端交互式对话

运行下面的脚本，程序会在命令行中和你的ChatBot进行交互式的对话，在命令行中输入指示并回车即可生成回复，输入 `clear` 可以清空对话历史，输入 `stop` 终止程序。

```bash
python cli_demo.py \
    --model_name_or_path ~/checkpoints/baichuan7b \ # base model
    --checkpoint_dir ./work_dir/checkpoint-700  \ # 训练的模型权重
    --trust_remote_code  \
    --double_quant \
    --quant_type nf4 \
    --fp16 \
    --bits 4
```

### 使用 Gradio 进行网页端交互

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `path/to/your/model_dir`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python gradio_webserver.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path  `path/to/your/model_dir`
```

## Sample Outputs

We provide generations for the models described in the paper for both OA and Vicuna queries in the `eval/generations` folder. These are intended to foster further research on model evaluation and analysis.

Can you distinguish ChatGPT from Guanaco? Give it a try!
You can access [the model response Colab here](https://colab.research.google.com/drive/1kK6xasHiav9nhiRUJjPMZb4fAED4qRHb?usp=sharing) comparing ChatGPT and Guanaco 65B on Vicuna prompts.

## Known Issues and Limitations

Here a list of known issues and bugs. If your issue is not reported here, please open a new issue and describe the problem.

1. 4-bit inference is slow. Currently, our 4-bit inference implementation is not yet integrated with the 4-bit matrix multiplication
2. Resuming a LoRA training run with the Trainer currently runs on an error
3. Currently, using `bnb_4bit_compute_type='fp16'` can lead to instabilities. For 7B LLaMA, only 80% of finetuning runs complete without error. We have solutions, but they are not integrated yet into bitsandbytes.
4. Make sure that `tokenizer.bos_token_id = 1` to avoid generation issues.

## License

`Efficient Finetuning of Quantized LLMs` is released under the Apache 2.0 license.

## Acknowledgements

We thank the Huggingface team, in particular Younes Belkada, for their support integrating QLoRA with PEFT and transformers libraries.

We appreciate the work by many open-source contributors, especially:

- [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LLaMa](https://github.com/facebookresearch/llama/)
- [Vicuna](https://github.com/lm-sys/FastChat/)
- [xTuring](https://github.com/stochasticai/xTuring)
- [Hugging Face](https://huggingface.co/)
- [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/)
- [Peft](https://github.com/huggingface/peft.git)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [qlora](https://github.com/artidoro/qlora)
- [deepspeed](https://www.deepspeed.ai/)
- [Unsloth](https://github.com/unslothai/unsloth)


## Citation

Please cite the repo if you use the data or code in this repo.

```bibtex
@misc{Chinese-Guanaco,
  author = {jianzhnie},
  title = {Chinese-Guanaco: Efficient Finetuning of Quantized LLMs for Chinese},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jianzhnie/Efficient-Tuning-LLMs}},
}
```
