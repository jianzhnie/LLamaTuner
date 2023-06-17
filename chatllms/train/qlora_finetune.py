import argparse
import copy
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import datasets
import torch
import transformers
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer,
                          PreTrainedTokenizer, Seq2SeqTrainer, Trainer,
                          set_seed)

from chatllms.utils.config import (DataArguments, GenerationArguments,
                                   LoraArguments, ModelArguments,
                                   QuantArgments, TrainingArguments)
from chatllms.utils.data_utils import (DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN,
                                       DEFAULT_PAD_TOKEN, DEFAULT_UNK_TOKEN,
                                       IGNORE_INDEX, make_data_module)
from chatllms.utils.callbacks import MMLUEvalCallback
from chatllms.utils.model_utils import (SavePeftModelCallback,
                                        find_all_linear_names,
                                        get_last_checkpoint,
                                        print_trainable_parameters,
                                        smart_tokenizer_and_embedding_resize,
                                        verify_dtypes)
from chatllms.utils.training import predict_and_save, train_and_evaluate

torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger(__name__)


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

    # Check if we are doing full finetuning.
    if args.full_finetune: assert args.bits in [16, 32]

    print(f'Loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else
                     (torch.bfloat16 if args.bf16 else torch.float32))
    torch_dtype = (torch.float32 if args.fp16 else
                   (torch.bfloat16 if args.bf16 else torch.float32))
    # Load the model.
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
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing)

    # Enable gradient checkpointing if specified.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            # Load pre-trained adapters from checkpoint directory.
            print('Loading adapters from checkpoint.')
            model = PeftModel.from_pretrained(model,
                                              os.path.join(
                                                  checkpoint_dir,
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


@dataclass
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

        Args:
            hf_dataset (dataset): The preprocesed dataset to load.
            tokenizer (PreTrainedTokenizer): The tokenizer to use when tokenizing the data.
            source_max_len (int): The maximum length allowed for the source text.
            target_max_len (int): The maximum length allowed for the target text.
            train_on_source (bool): If True, the model will be trained on the source text as well as the target text.
            predict_with_generate (bool): If True, the model will generate predictions instead of training.
    """
    def __init__(
        self,
        hf_dataset: datasets.DatasetDict,
        tokenizer: PreTrainedTokenizer,
        source_max_len: int,
        target_max_len: int,
        train_on_source: bool,
        predict_with_generate: bool = False,
    ):

        super(SupervisedDataset, self).__init__()
        # Load the dataset and format it
        logging.warning('Loading data...')
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return an item from the dataset based on its index."""
        example = self.dataset[idx]
        # Tokenize the source text
        source_txt = f"{self.tokenizer.bos_token}{example['input']}"
        tokenized_source = self.tokenizer(
            source_txt,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Tokenize the target text
        target_txt = f"{example['output']}{self.tokenizer.eos_token}"
        tokenized_target = self.tokenizer(
            target_txt,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        src_ids = tokenized_source['input_ids']
        tgt_ids = tokenized_target['input_ids']
        if not self.predict_with_generate:
            # If not generating predictions, concatenate the input and target ids
            input_ids = torch.tensor(src_ids + tgt_ids)
            if not self.train_on_source:
                # If not training on the source text, set the labels to IGNORE_INDEX \
                # for the input ids and the target ids
                labels = torch.tensor(
                    [IGNORE_INDEX
                     for _ in range(len(src_ids))] + copy.deepcopy(tgt_ids))
            else:
                # If training on the source text, set the labels to the concatenated \
                # input and target ids
                labels = torch.tensor(copy.deepcopy(src_ids + tgt_ids))
        else:
            # If generating predictions, only use the source ids as input
            input_ids = torch.tensor(src_ids)
            labels = None

        # Construct data dictionary containing inputs and labels
        data_dict = {'input_ids': input_ids, 'labels': labels}

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """
    Collate examples for supervised fine-tuning.

    Args:
        tokenizer (PreTrainedTokenizer): The pre-trained tokenizer to use.
        predict_with_generate (bool): Whether to do prediction with generate or not.
    """

    tokenizer: PreTrainedTokenizer
    predict_with_generate: bool = False

    def __call__(
            self,
            instances: List[Dict[str,
                                 torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples for supervised fine-tuning on a sequence classification task \
            using a pre-trained tokenizer.

        Args:
            instances (List[Dict[str, torch.Tensor]]): A list of dictionaries containing the keys\
                  'input_ids' and 'labels'.

        Returns:
            A dictionary containing the collated batch with keys 'input_ids', 'labels', and 'attention_mask'.
        """

        # Extract input IDs and labels from each instance
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))

        # Pad sequences to be of equal length
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        ) if not self.predict_with_generate else None

        # Construct attention mask based on padded input IDs
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Return collated batch as dictionary
        data_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict


def main():
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
        extra_args,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
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
    train_dataset = SupervisedDataset(
        dataset_dict['train'],
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    ) if args.do_train else None

    eval_dataset = SupervisedDataset(
        dataset_dict['eval'],
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    ) if args.do_eval else None

    predict_dataset = SupervisedDataset(
        dataset_dict['predict'],
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    ) if args.do_predict else None

    print(train_dataset, eval_dataset, predict_dataset)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, predict_with_generate=args.predict_with_generate)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    # Add callback to save adapter model.
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        eval_callback = MMLUEvalCallback(
            trainer=trainer,
            tokenizer=tokenizer,
            data_dir='./data',
            args=args,
        )
        trainer.add_callback(eval_callback)

    # Verify dtypes
    verify_dtypes(model)
    assert args.do_train or args.do_eval or args.do_predict
    if args.do_train or args.do_eval:
        train_and_evaluate(trainer, args)
    if args.do_predict:
        predict_and_save(trainer, tokenizer, predict_dataset, args)


if __name__ == '__main__':
    main()
