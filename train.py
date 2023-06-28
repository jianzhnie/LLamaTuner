import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, PreTrainedModel,
                          PreTrainedTokenizer, Trainer, TrainingArguments)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={'help': 'Path to the training data.'})


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    model_max_length: int = field(
        default=2048,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict,
                                         tokenizer: PreTrainedTokenizer,
                                         model: PreTrainedModel):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

    Attributes:
        PROMPT_DICT (dict): A dictionary containing prompts for the model to complete.
        IGNORE_INDEX (int): A value to replace tokens corresponding to the source in the labels tensor.

    Methods:
        __init__(self, data_path: str, tokenizer: PreTrainedTokenizer): Initializes a SupervisedDataset object.
        __len__(self) -> int: Returns the length of the dataset.
        __getitem__(self, idx) -> Dict[str, torch.Tensor]: Retrieves an example from the dataset at the specified index.

    """

    PROMPT_DICT = {
        'prompt_input':
        ('Below is an instruction that describes a task, paired with an input that provides further context. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
         ),
        'prompt_no_input':
        ('Below is an instruction that describes a task. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Response:'),
    }
    IGNORE_INDEX = -100

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024):
        """
        Initializes a SupervisedDataset object.

        Args:
            data_path (str): The path to the training data file.
            tokenizer (PreTrainedTokenizer): The tokenizer object used to tokenize the input examples.

        """
        super(SupervisedDataset, self).__init__()
        logging.warning(f'Loading dataset from {data_path}')
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = load_dataset('json',
                                          data_files=data_path)['train']
        else:
            list_data_dict = load_dataset(data_path)['train']

        logging.warning('Found %d rows', list_data_dict.num_rows)
        prompt_input, prompt_no_input = self.PROMPT_DICT[
            'prompt_input'], self.PROMPT_DICT['prompt_no_input']
        self.sources = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        self.targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        self.examples = [s + t for s, t in zip(self.sources, self.targets)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of examples in the dataset.

        """
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Retrieves an example from the dataset at the specified index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the input_ids, labels, input_len, source_input_ids, and
            source_len tensors.

        """
        example_txt = self.examples[idx]
        # Tokenize the example and source text
        example_tokenized = self.tokenizer(
            example_txt,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
        )
        source_txt = self.sources[idx]
        source_tokenized = self.tokenizer(
            source_txt,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
        )
        # Extract the input_ids tensor
        input_ids = torch.tensor(example_tokenized['input_ids'])
        # Create the labels tensor
        labels = input_ids.clone()
        labels[:len(source_tokenized['input_ids'])] = self.IGNORE_INDEX
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels,
                              batch_first=True,
                              padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train() -> None:
    """
    Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.

    Returns:
        None

    """
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load the pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side='right',
        use_fast=False,
    )

    # Resize the tokenizer's vocabulary size to accommodate additional special tokens, if necessary
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    special_tokens_dict['additional_special_tokens'] = [
        '### Instruction:',
        '### Response:\n',
        '### End',
    ]

    if len(special_tokens_dict) > 0:
        smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer,
                                             model)

    max_length = None
    for length_setting in [
            'n_positions', 'max_position_embeddings', 'seq_length'
    ]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logging.warning(f'Found max lenth: {max_length}')
            break
    if not max_length:
        max_length = 1024
        logging.warning(f'Using default max length: {max_length}')

    # Create the training dataset and data collator
    train_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    model.is_parallelizable = True
    model.model_parallel = True

    # Initialize the Trainer object and start training
    logging.warning('Instantiating Trainer')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    logging.warning('Training')

    if training_args.resume_from_checkpoint and list(
            pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    logging.warning(f'Saving Model to {training_args.output_dir}')
    trainer.save_state()
    # Save the trained model
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)

    logging.warning('Done.')


if __name__ == '__main__':
    train()
