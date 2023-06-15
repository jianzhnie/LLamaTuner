import argparse
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from os.path import exists, isdir, join
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer,
                          PreTrainedTokenizer, Seq2SeqTrainer, set_seed)

from utils.data_utils import format_dataset, load_data, split_train_eval
from utils.model_utils import (SavePeftModelCallback, find_all_linear_names,
                               print_trainable_parameters,
                               smart_tokenizer_and_embedding_resize,
                               verify_dtypes)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'

torch.backends.cuda.matmul.allow_tf32 = True
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
    dataset_name: str = field(
        default='alpaca',
        metadata={
            'help': 'Which dataset to finetune on. See datamodule for options.'
        })
    eval_dataset_size: int = field(
        default=1024, metadata={'help': 'Size of validation dataset.'})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'For debugging purposes or quicker training, truncate the number of training examples to this '
            'value if set.'
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'For debugging purposes or quicker training, truncate the number of evaluation examples to this '
            'value if set.'
        },
    )
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
    do_train: bool = field(
        default=True,
        metadata={'help': 'To train or not to train, that is the question?'})
    do_eval: bool = field(
        default=False,
        metadata={'help': 'To train or not to train, that is the question?'})
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Whether to train on the input in addition to the target text.'
        })
    optim: str = field(default='paged_adamw_32bit',
                       metadata={'help': 'The optimizer to be used'})
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            'help':
            'Gradient clipping max norm. This is tuned and works well for all models tested.'
        })
    gradient_checkpointing: bool = field(
        default=True,
        metadata={'help': 'Use gradient checkpointing. You want to use this.'})
    group_by_length: bool = field(
        default=True,
        metadata={
            'help':
            'Group sequences into batches with same length. Saves memory and speeds up training considerably.'
        })
    predict_with_generate: bool = field(
        default=False,
        metadata={
            'help':
            'Group sequences into batches with same length. Saves memory and speeds up training considerably.'
        })


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

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def get_accelerate_model(args: Dict,
                         checkpoint_dir: Optional[str]) -> torch.nn.Module:
    """
    Returns a language model for text generation that can be trained with mixed precision.

    Args:
        args (Dict): A dictionary containing various hyperparameters.
        checkpoint_dir (str, optional): A directory containing pre-trained adapters for the model.

    Returns:
        torch.nn.Module: An instance of the language model.
    """
    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = 'auto'

    # If we are in a distributed setting, we need to set the device map and max memory per device.
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    print(f'Loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else
                     (torch.bfloat16 if args.bf16 else torch.float32))
    torch_dtype = (torch.float32 if args.fp16 else
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
        torch_dtype=torch_dtype,
        use_auth_token=args.use_auth_token)

    # Print a message if the GPU supports bfloat16.
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('=' * 80)
            print(
                'Your GPU supports bfloat16, you can accelerate training with the argument --bf16'
            )
            print('=' * 80)

    # Enable model parallelism.
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = torch_dtype

    # Prepare the model for k-bit training if specified.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing)

    # Enable gradient checkpointing if specified.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        # Load pre-trained adapters from checkpoint directory.
        print('Loading adapters from checkpoint.')
        model = PeftModel.from_pretrained(model,
                                          os.path.join(checkpoint_dir,
                                                       'adapter_model'),
                                          is_trainable=True)
    else:
        # Add LoRA modules to the model.
        print('Adding LoRA modules...')
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

    # Convert certain model modules to a different precision as specified by the hyperparameters.
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


class DataCollatorForCausalLM(object):
    """
    Data collator used for language modeling tasks. This collator takes in a sequence of examples
    (input/output pairs) and returns a dictionary containing the inputs and labels for training
    a causal language model.

    Parameters:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to tokenize the input and output text.
        source_max_len (int): The maximum length allowed for the input source text.
        target_max_len (int): The maximum length allowed for the target output text.
        train_on_source (bool): If True, the model will be trained on the source text. Otherwise, it will be trained
                                on both source and target text concatenated together.
        predict_with_generate (bool, default=False): If True, only the input_ids for the tokenized source text
                                                      are returned. This is useful during inference when generating
                                                      text sequences from the model.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        source_max_len: int,
        target_max_len: int,
        train_on_source: bool,
        predict_with_generate: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate

    def __call__(
            self, instances: Sequence[Dict[str,
                                           str]]) -> Dict[str, torch.Tensor]:
        """
        Takes a sequence of input/output pairs and returns a dictionary containing the inputs and labels
        for training a causal language model.

        Parameters:
            instances (Sequence[Dict[str, str]]): A sequence of input/output pairs. Each dictionary must contain
                                                  the keys 'input' and 'output'.

        Returns:
            data_dict (Dict[str, torch.Tensor]): A dictionary containing the input_ids, attention_mask,
                                                 and optionally the labels.
        """
        # Extract elements
        sources: List[str] = [
            f"{self.tokenizer.bos_token}{example['input']}"
            for example in instances
        ]
        targets: List[str] = [
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
                    # train_on_source 默认设置为 False, 训练时不在 source  text 上计算损失
                    labels.append(
                        torch.tensor([
                            IGNORE_INDEX for _ in range(len(tokenized_source))
                        ] + copy.deepcopy(tokenized_target)))
                else:
                    # 如果 train_on_source 设置为 True, 训练时将 source text  和 target text 的标签合并, 然后计算损失
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
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        ) if not self.predict_with_generate else None

        # Construct data dictionary containing inputs and labels
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict


def get_last_checkpoint(checkpoint_dir: str) -> Tuple[str, bool]:
    """
    Given a directory containing previous saved checkpoints, returns the path to the last checkpoint
    if available along with a boolean flag indicating whether training has already been completed.

    Args:
        checkpoint_dir (str): Path to the directory containing the saved checkpoints.

    Returns:
        A tuple containing the path to the last checkpoint if available, and a boolean flag indicating
        whether training has already been completed.
    """
    # Check if provided directory exists
    if isdir(checkpoint_dir):

        # Check if 'completed' file exists in the directory - indicates training has completed
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed:
            return None, True  # Already finished

        # Find the latest checkpoint by checking all subdirectories named 'checkpoint-*'
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir,
                          filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step,
                               int(filename.replace('checkpoint-', '')))
        if max_step == 0:
            return None, is_completed  # Training started, but no checkpoint found

        # Return path to the latest checkpoint directory
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f'Found a previous checkpoint at: {checkpoint_dir}')
        return checkpoint_dir, is_completed

    # The directory does not exist, meaning this is the first time the training is being run
    return None, False


def make_data_module(args):
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    dataset = load_data(args.dataset_name)
    dataset = format_dataset(dataset, dataset_name=args.dataset_name)
    dataset_dict = split_train_eval(
        dataset,
        do_eval=args.do_eval,
        eval_dataset_size=args.eval_dataset_size,
        max_eval_samples=args.max_eval_samples,
        group_by_length=args.group_by_length,
        do_train=args.do_train,
        max_train_samples=args.max_train_samples,
    )

    return dataset_dict


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
    print(args)
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

    dataset_dict = make_data_module(args)
    print(dataset_dict)
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
        train_dataset=dataset_dict['train'] if args.do_train else None,
        eval_dataset=dataset_dict['eval'] if args.do_eval else None,
        data_collator=data_collator,
    )
    trainer.add_callback(SavePeftModelCallback)

    # Verify dtypes
    verify_dtypes(model)
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
