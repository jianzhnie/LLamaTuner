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

👋 Join our [WeChat](assets/wechat.jpeg).

# Efficient Finetuning of Quantized LLMs  --- 低资源的大语言模型量化训练/部署方案

This is the repo for the `Efficient Finetuning of Quantized LLMs` project, which aims to build and share instruction-following Chinese `baichuan-7b/LLaMA/Pythia/GLM`model tuning methods which can be trained on **a single Nvidia RTX-2080TI**, multi-round chatbot which can be trained on **a single Nvidia RTX-3090** with the context len 2048.

We uses [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization and is integrated with Huggingface's [PEFT](https://github.com/huggingface/peft) and [transformers](https://github.com/huggingface/transformers/) libraries.

The repo contains:
- code for finetune the model
- code for generation based on trained model
- code for run on CPU (fp16 or int4 is support, in purely C++)


## Overview

We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. We release all of our models and code, including CUDA kernels for 4-bit training.


## Installation
To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source and make sure you have the latest version of the bitsandbytes library (0.39.0). You can achieve the above with the following commands:
```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

## Getting Started

## QLora int8 Finetune
```bash
python chatllms/train/qlora_int8_finetune.py \
    --model_name_or_path  decapoda-research/llama-7b-hf  \
    --data_path tatsu-lab/alpaca  \
    --output_dir work_dir_lora/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 2048 \
    --logging_steps 1 \
    --fp16 True
```

## QLora int4 Finetune

The `qlora_int4_finetune.py` code is a starting point for finetuning and inference on various datasets.
Basic command for finetuning a baseline model on the Alpaca dataset:
```bash
python chatllms/train/qlora_int4_finetune.py --model_name_or_path <path_or_name>
```

For models larger than 13B, we recommend adjusting the learning rate:
```bash
python chatllms/train/qlora_int4_finetune.py –learning_rate 0.0001 --model_name_or_path <path_or_name>
```

We can also tweak our hyperparameters:

```bash
python chatllms/train/qlora_int4_finetune.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/guanaco-7b \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
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
python chatllms/train/qlora_int4_finetune.py --dataset="path/to/your/dataset"
```
- Training with a self-instruct format dataset:

```python
python chatllms/train/qlora_int4_finetune.py --dataset="path/to/your/dataset" --dataset_format="self-instruct"
```

## Multi GPU
Multi GPU training and inference work out-of-the-box with Hugging Face's Accelerate. Note that the per_device_train_batch_size and per_device_eval_batch_size arguments are global batch sizes unlike what their name suggest.

When loading a model for training or inference on multiple GPUs you should pass something like the following to AutoModelForCausalLM.from_pretrained():
```python
device_map = "auto"
max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
```

## Inference

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `path/to/your/model_dir`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python chatllms/server/gradio_webserver.py \
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

- [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/)
- [LoRA](https://github.com/microsoft/LoRA/)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/)
- [Hugging Face](https://huggingface.co/)
- [LLaMa](https://github.com/facebookresearch/llama/)
- [Vicuna](https://github.com/lm-sys/FastChat/)


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
