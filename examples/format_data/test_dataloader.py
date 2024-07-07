import os
import sys

sys.path.append(os.getcwd())
from transformers import AutoTokenizer, HfArgumentParser

from llamatuner.configs import (DataArguments, FinetuningArguments,
                                ModelArguments)
from llamatuner.data.data_loader import load_single_dataset
from llamatuner.data.data_parser import get_dataset_list

if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, FinetuningArguments))
    (model_args, data_args,
     training_args) = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset_list = get_dataset_list(data_args)
    dataset = load_single_dataset(dataset_list[0], model_args, data_args)
    print(dataset[:2])
