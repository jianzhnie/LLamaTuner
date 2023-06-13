import argparse
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from os.path import exists, isdir, join
from typing import Any, Dict, Optional, Sequence

import torch
import transformers
from datasets import load_dataset
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer, Seq2SeqTrainer,
                          set_seed)

from utils.model_utils import (SavePeftModelCallback, find_all_linear_names,
                               print_trainable_parameters,
                               smart_tokenizer_and_embedding_resize)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained.'
        })
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Enables using Huggingface auth token from Git Credentials.'
        })


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={'help': 'Path to the training data.'})
    source_max_len: int = field(
        default=1024,
        metadata={
            'help':
            'Maximum source sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    target_max_len: int = field(
        default=256,
        metadata={
            'help':
            'Maximum target sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            'help':
            'Gradient clipping max norm. This is tuned and works well for all models tested.'
        })
    gradient_checkpointing: bool = field(
        default=True,
        metadata={'help': 'Use gradient checkpointing. You want to use this.'})


@dataclass
class LoraArguments:
    lora_r: int = field(default=64, metadata={'help': 'Lora R dimension.'})
    lora_alpha: float = field(default=16, metadata={'help': ' Lora alpha.'})
    lora_dropout: float = field(default=0.0,
                                metadata={'help': 'Lora dropout.'})
    max_memory_MB: int = field(default=80000,
                               metadata={'help': 'Free memory per gpu.'})
    lora_weight_path: str = ''
    bias: str = 'none'


@dataclass
class QuantArgments:
    adam8bit: bool = field(default=False, metadata={'help': 'Use 8-bit adam.'})
    double_quant: bool = field(
        default=True,
        metadata={
            'help':
            'Compress the quantization statistics through double quantization.'
        })
    quant_type: str = field(
        default='nf4',
        metadata={
            'help':
            'Quantization data type to use. Should be one of `fp4` or `nf4`.'
        })
    bits: int = field(default=4, metadata={'help': 'How many bits to use.'})


@dataclass
class GenerationArguments:
    # generation parameters
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            'help':
            'Maximum number of new tokens to be generated in evaluation or prediction loops'
            'if predict_with_generate is set.'
        })
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'Minimum number of new tokens to generate.'})

    # gebneration strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # hyperparameters for logits processing
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = 'auto'

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else
                     (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type  # {'fp4', 'nf4'}
        ),
        torch_dtype=(torch.float32 if args.fp16 else
                     (torch.bfloat16 if args.bf16 else torch.float32)),
        use_auth_token=args.use_auth_token)
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('=' * 80)
            print(
                'Your GPU supports bfloat16, you can accelerate training with the argument --bf16'
            )
            print('=' * 80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = (torch.float32 if args.fp16 else (
        torch.bfloat16 if args.bf16 else torch.float32))

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        print('Loading adapters from checkpoint.')
        model = PeftModel.from_pretrained(model,
                                          join(checkpoint_dir,
                                               'adapter_model'),
                                          is_trainable=True)
    else:
        print('adding LoRA modules...')
        modules = find_all_linear_names(args, model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}"
            for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}"
            for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']):
            if not self.predict_with_generate:
                input_ids.append(
                    torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([
                            IGNORE_INDEX for _ in range(len(tokenized_source))
                        ] + copy.deepcopy(tokenized_target)))
                else:
                    labels.append(
                        torch.tensor(
                            copy.deepcopy(tokenized_source +
                                          tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        ) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


def load_and_format_dataset(data_path):

    ALPACA_PROMPT_DICT = {
        'prompt_input':
        ('Below is an instruction that describes a task, paired with an input that provides further context. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: '
         ),
        'prompt_no_input':
        ('Below is an instruction that describes a task. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Response: '),
    }

    def extract_alpaca_dataset(example):
        if example.get('input', '') != '':
            prompt_format = ALPACA_PROMPT_DICT['prompt_input']
        else:
            prompt_format = ALPACA_PROMPT_DICT['prompt_no_input']
        return {'input': prompt_format.format(**example)}

    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path)['train']
    else:
        dataset = load_dataset(data_path)['train']

    dataset = dataset.map(extract_alpaca_dataset,
                          remove_columns=['instruction'])
    # Remove unused columns.
    dataset = dataset.remove_columns([
        col for col in dataset.column_names if col not in ['input', 'output']
    ])
    return dataset


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir,
                          filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step,
                               int(filename.replace('checkpoint-', '')))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f'Found a previous checkpoint at: {checkpoint_dir}')
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments,
         QuantArgments, GenerationArguments))
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        quant_args,
        generation_args,
    ) = parser.parse_args_into_dataclasses()
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args))

    args = argparse.Namespace(**vars(model_args), **vars(data_args),
                              **vars(training_args), **vars(lora_args),
                              **vars(quant_args))

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False
    print_trainable_parameters(args, model)
    print('loaded model')
    set_seed(args.seed)

    # Tokenizer
    if model.config.model_type == 'llama':
        # Due to the name of Transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            padding_side='right',
            use_fast=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            padding_side='right',
            use_fast=False,  # Fast tokenizer giving issues.
            tokenizer_type='llama' if 'llama' in args.model_name_or_path else
            None,  # Needed for HF name change
            use_auth_token=args.use_auth_token,
        )
    if 'llama' in args.model_name_or_path or isinstance(
            tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        special_tokens_dict: Dict[str, Any] = {}
        if tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

        if len(special_tokens_dict) > 0:
            smart_tokenizer_and_embedding_resize(special_tokens_dict,
                                                 tokenizer, model)

    dataset = load_and_format_dataset(args.data_path)

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {'run_name': args.run_name}
    # Training
    logger.info('*** Train ***')
    # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
    # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    all_metrics.update(metrics)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fout:
        fout.write(json.dumps(all_metrics))


if __name__ == '__main__':
    train()
