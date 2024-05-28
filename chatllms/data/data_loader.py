import argparse
import logging

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from chatllms.data.conv_dataset import ConversationDataset, VicunaDataset
from chatllms.data.data_utils import make_data_module
from chatllms.data.sft_dataset import (DataCollatorForSupervisedDataset,
                                       SupervisedDataset)


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    args: argparse.Namespace,
    text_logger: logging.Logger,
) -> dict[str, torch.utils.data.Dataset]:
    train_dataset, eval_dataset, multi_turn = make_data_module(
        args, text_logger)
    max_seq_length = tokenizer.model_max_length
    dataset_cls = (VicunaDataset if args.conversation_template == 'vicnua' else
                   ConversationDataset)

    if not multi_turn:
        train_dataset = (SupervisedDataset(
            train_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length,
        ) if args.do_train else None)

        eval_dataset = (SupervisedDataset(
            eval_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length,
        ) if args.do_eval else None)

    else:
        train_dataset = dataset_cls(
            train_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        ) if args.do_train else None
        eval_dataset = dataset_cls(
            eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        ) if args.do_eval else None

    if args.do_train:
        train_info = f'train_dataset: {type(train_dataset)}, mutlti-turn: {multi_turn},  #length: {len(train_dataset)}'
        text_logger.info(train_info)

    if args.do_eval:
        eval_info = f'eval_dataset: {type(eval_dataset)}, mutlti-turn: {multi_turn}, #length: {len(eval_dataset)}'
        text_logger.info(eval_info)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, predict_with_generate=args.predict_with_generate)

    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
