import json
import sys
from typing import Dict, List, Sequence

from transformers import PreTrainedTokenizer

sys.path.append('../')
import transformers

from chatllms.data.data_utils import IGNORE_INDEX as IGNORE_TOKEN_ID
from chatllms.data.utils.conversation import Conversation, SeparatorStyle


def preprocess(
    sources: Sequence[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dict[str, List[int]]:
    """
    Preprocesses the data by tokenizing it.

    Args:
        sources (Sequence[Dict[str, str]]): List of conversation sources.
            Each source is a dictionary containing 'from' (sender role) and 'value' (message content).
        tokenizer (PreTrainedTokenizer): Tokenizer for tokenizing the conversations.

    Returns:
        Dict[str, List[int]]: A dictionary containing the preprocessed data.
            - 'input_ids': Tokenized input conversation IDs.
            - 'labels': Tokenized target conversation IDs.
            - 'attention_mask': Attention mask for the input conversation.

    Raises:
        AssertionError: If the roles in the conversation are not consistent.

    """

    # Create a Conversation object
    conv = Conversation(
        name='vicuna_v1.1',
        system=
        'A chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=['USER', 'ASSISTANT'],
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=' ',
        sep2='</s>',
    )
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first message if it is not from the human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Check if the roles are consistent
    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for rou in rounds:
            if rou == '':
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            tgt_txt = tokenizer.decode(target[cur_len:cur_len +
                                              instruction_len])
            print(tgt_txt)
            target[cur_len:cur_len + instruction_len] = IGNORE_TOKEN_ID
            print('*' * 20)
            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' (ignored)')

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


if __name__ == '__main__':
    data_path = '/home/robin/work_dir/llm/Chinese-Guanaco/examples/sharegpt_formate_role_filter.json'
    # Load the raw data from the specified data_path
    data_path = '/home/robin/work_dir/llm/FastChat/data/dummy_conversation.json'
    with open(data_path, 'r') as file:
        raw_data = json.load(file)

    sources = [example['conversations'] for example in raw_data]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'facebook/opt-125m',
        model_max_length=512,
        padding_side='right',
        use_fast=False,
    )

    data = preprocess([sources[0]], tokenizer=tokenizer)
    print(data)
