from transformers.tokenization_utils import PreTrainedTokenizer

from .conv_dataset import ConversationDataset, VicunaDataset
from .data_utils import make_data_module
from .sft_dataset import (DataCollatorForSupervisedDataset,
                          SFTInstructionDataset)


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, args):
    train_dataset, eval_dataset, multi_turn = make_data_module(args)
    max_length = tokenizer.model_max_length
    dataset_cls = (VicunaDataset if args.vicuna_conversation_formate else
                   ConversationDataset)

    if not multi_turn:
        train_dataset = SFTInstructionDataset(
            train_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_length,
        ) if args.do_train else None

        eval_dataset = SFTInstructionDataset(
            train_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_length,
        ) if args.do_eval else None

    else:
        train_dataset = dataset_cls(
            train_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_length,
        ) if args.do_train else None
        eval_dataset = dataset_cls(
            eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_length,
        ) if args.do_eval else None

    print(
        f'train_dataset: {type(train_dataset)}, mutlti-turn: {multi_turn},  #length: {len(train_dataset)}'
    ) if args.do_train else None
    print(
        f'eval_dataset: {type(eval_dataset)},mutlti-turn: {multi_turn}, #length: {len(eval_dataset)}'
    ) if args.do_eval else None

    print('Adding data collator: ', DataCollatorForSupervisedDataset)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, predict_with_generate=args.predict_with_generate)

    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
