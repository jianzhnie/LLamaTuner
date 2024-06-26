from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from transformers import PreTrainedTokenizer

from llamatuner.data.formatter import (EmptyFormatter, Formatter,
                                       FunctionFormatter, StringFormatter,
                                       ToolFormatter)
from llamatuner.data.utils import Role, infer_max_len
from llamatuner.utils.logger_utils import get_logger

logger = get_logger('llamatuner')

SYSTEM_TEMPLATE = dict(
    alpaca=('Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n'),
    coder=('You are a professional programer. Please provide the '
           'corresponding code based on the description of Human.\n'),
    lawyer='你现在是一名专业的中国律师，请根据用户的问题给出准确、有理有据的回复。\n',
    medical='如果你是一名医生，请根据患者的描述回答医学问题。\n',
    sql=('If you are an expert in SQL, please generate a good SQL Query '
         'for Question based on the CREATE TABLE statement.\n'),
)

DEFAULT_PROMPT_DICT = {
    'prompt_input': ('{instruction}{input}'),
    'prompt_no_input': ('{instruction}'),
}

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

RANDOM_PROMPT_DICT = {
    'prompt_input': [
        # input encoding template, output encoding template, weight
        ('{instruction}\n\n{input}\n\n', 0.2),
        ('{instruction}\n{input}\n\n', 0.1),
        ('{instruction}\n{input}\n', 0.1),
        ('{instruction}\n\nInput: {input}\n\nOutput:', 0.05),
        ('{instruction}\nInput: {input}\nOutput:', 0.05),
        ('{instruction}\n{input}\n\nResponse:', 0.05),
        ('{instruction}\n\nAdditional Context:\n{input}\n\nAnswer:', 0.05),
        ('Task: {instruction}\nInput: {input}\nOutput:', 0.05),
        ('Task: {instruction}\n\n{input}\n\n', 0.05),
        ('Task: {instruction}\n\n{input}\n\nAnswer:', 0.05),
        (
            'You need to complete the following task:\n\n{instruction}\n\n{input}\n\nAnswer:',
            0.05,
        ),
        (
            '{instruction}\n\nNow complete the following instance -\nInput: {input}\nOutput:',
            0.05,
        ),
        ('Instruction:{instruction}\n\nInput: {input}\n\n', 0.05),
        (
            'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'
            '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: ',
            0.1,
        ),  # alpaca template
    ],
    'prompt_no_input': [
        ('{instruction}\n\n', 0.2),
        ('{instruction}\n', 0.1),
        ('{instruction}\n\nOutput:', 0.1),
        ('{instruction}\nOutput:', 0.05),
        ('{instruction}\nResponse:', 0.05),
        ('{instruction}\n\nAnswer:', 0.05),
        ('Task: {instruction}\n\n', 0.05),
        ('Instruction: {instruction}\n', 0.05),
        ('Instruction: {instruction}\nOutput:', 0.05),
        ('You need to complete the following task:\n\n{instruction}\n\n',
         0.05),
        ('Can you help with this?\n\n{instruction}\n', 0.05),
        ('Plase answer the following request: {instruction}\nAnswer:', 0.05),
        (
            'Tell me how would you respond to the following request.\n{instruction}\n',
            0.05,
        ),
        (
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:',
            0.1,
        ),  # alpaca template
    ],
}


@dataclass
class Template:
    """
    Template 类的主要作用是将对话消息（如用户和助手之间的对话）格式化并编码成 token IDs，\
    这些 token IDs 可以用于自然语言处理（NLP）任务中的模型输入。
    这个类提供了一些方法来处理单轮和多轮对话的编码

    Args:
        - format_user：用户消息的格式化器。
        - format_assistant：助手消息的格式化器。
        - format_system：系统消息的格式化器。
        - format_function：功能消息的格式化器。
        - format_observation：观察消息的格式化器。
        - format_tools：工具信息的格式化器。
        - format_separator：消息之间分隔符的格式化器。
        - default_system：默认的系统消息，如果未提供系统消息时使用。
        - stop_words：处理过程中使用的停用词列表。
        - image_token：图像标记。
        - efficient_eos：高效处理结束标记的标志。
        - replace_eos：替换结束标记的标志。
        - force_system：强制包含系统消息的标志。
    """

    format_user: Formatter
    format_assistant: Formatter
    format_system: Formatter
    format_function: Formatter
    format_observation: Formatter
    format_tools: Formatter
    format_separator: Formatter
    default_system: str
    stop_words: List[str]
    image_token: str
    efficient_eos: bool
    replace_eos: bool
    force_system: bool

    def encode_oneturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Tuple[List[int], List[int]]:
        """
        这个方法用于编码单轮对话，将其表示为提示和响应的 token IDs 序列。

        Returns a single pair of token IDs representing prompt and response respectively.

        方法逻辑:

            - 调用 _encode 方法：首先，调用 _encode 方法对消息进行编码，生成一个包含成对的 token IDs 的序列。
            - 拼接 token IDs：接着，将所有消息的查询部分（query_ids）和响应部分（resp_ids）拼 \
              接起来，生成一个完整的提示序列 prompt_ids。最后一个查询部分的 token IDs 和最后一个响应\
              部分的 token IDs 分别作为提示和响应返回。

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to convert text to tokens.
            messages (List[Dict[str, str]]): List of message dictionaries containing roles and content.
            system (Optional[str]): System message to include at the beginning.
            tools (Optional[str]): Tools information to include.
            cutoff_len (int): Maximum allowed length for the encoded sequences.
            reserved_label_len (int): Length reserved for the label.

        Returns:

            返回一个元组，包含两个列表：提示（prompt_ids）和响应（answer_ids）的 token IDs。
            Tuple[List[int], List[int]]: Encoded prompt and response token IDs.
        """
        encoded_pairs = self._encode(tokenizer, messages, system, tools,
                                     cutoff_len, reserved_label_len)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids += query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        answer_ids = encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        """
        Returns multiple pairs of token IDs representing prompts and responses respectively.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to convert text to tokens.
            messages (List[Dict[str, str]]): List of message dictionaries containing roles and content.
            system (Optional[str]): System message to include at the beginning.
            tools (Optional[str]): Tools information to include.
            cutoff_len (int): Maximum allowed length for the encoded sequences.
            reserved_label_len (int): Length reserved for the label.

        Returns:
            Sequence[Tuple[List[int], List[int]]]: Encoded prompt and response token ID pairs.
        """
        return self._encode(tokenizer, messages, system, tools, cutoff_len,
                            reserved_label_len)

    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        """
        Encodes formatted inputs to pairs of token IDs.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to convert text to tokens.
            messages (List[Dict[str, str]]): List of message dictionaries containing roles and content.
            system (Optional[str]): System message to include at the beginning.
            tools (Optional[str]): Tools information to include.
            cutoff_len (int): Maximum allowed length for the encoded sequences.
            reserved_label_len (int): Length reserved for the label.

        主要步骤：
        - 初始化系统消息：使用提供的系统消息或默认系统消息。
        - 处理每条消息：遍历消息列表，根据消息的角色（用户、助手等）应用相应的格式化器。
        - 转换为 token IDs：将格式化后的元素转换为 token IDs。
        - 生成成对序列：将所有的 token IDs 按查询和响应成对分组，返回这些成对的序列。

        Returns:
            Sequence[Tuple[List[int], List[int]]]: Encoded prompt and response token ID pairs.
        """
        system = system or self.default_system
        encoded_messages = []

        for i, message in enumerate(messages):
            elements = []
            if i == 0 and (system or tools or self.force_system):
                tool_text = self.format_tools.apply(
                    content=tools)[0] if tools else ''
                elements += self.format_system.apply(content=(system +
                                                              tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message['role'] == Role.USER:
                elements += self.format_user.apply(content=message['content'],
                                                   idx=str(i // 2))
            elif message['role'] == Role.ASSISTANT:
                elements += self.format_assistant.apply(
                    content=message['content'])
            elif message['role'] == Role.OBSERVATION:
                elements += self.format_observation.apply(
                    content=message['content'])
            elif message['role'] == Role.FUNCTION:
                elements += self.format_function.apply(
                    content=message['content'])
            else:
                raise NotImplementedError(
                    f'Unexpected role: {message["role"]}')

            encoded_messages.append(
                self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len,
                                reserved_label_len)

    def _convert_elements_to_ids(
        self,
        tokenizer: PreTrainedTokenizer,
        elements: List[Union[str, Dict[str, str]]],
    ) -> List[int]:
        """
        Converts elements to token IDs.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to convert text to tokens.
            elements (List[Union[str, Dict[str, str]]]): List of elements to convert.

        Returns:
            List[int]: List of token IDs.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if elem:
                    token_ids += tokenizer.encode(elem,
                                                  add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids.append(
                    tokenizer.convert_tokens_to_ids(elem.get('token')))
            elif isinstance(elem, set):
                if 'bos_token' in elem and tokenizer.bos_token_id is not None:
                    token_ids.append(tokenizer.bos_token_id)
                elif 'eos_token' in elem and tokenizer.eos_token_id is not None:
                    token_ids.append(tokenizer.eos_token_id)
            else:
                raise ValueError(
                    f'Input must be string, set[str] or dict[str, str], got {type(elem)}'
                )
        return token_ids

    def _make_pairs(
        self,
        encoded_messages: Sequence[List[int]],
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        """
        Creates pairs of source and target token IDs.

        Args:
            encoded_messages (Sequence[List[int]]): List of encoded messages.
            cutoff_len (int): Maximum allowed length for the encoded sequences.
            reserved_label_len (int): Length reserved for the label.

        Returns:
            Sequence[Tuple[List[int], List[int]]]: Sequence of source and target token ID pairs.
        """
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            max_source_len, max_target_len = infer_max_len(
                source_len=len(encoded_messages[i]),
                target_len=len(encoded_messages[i + 1]),
                max_len=(cutoff_len - total_length),
                reserved_label_len=reserved_label_len,
            )
            source_ids = encoded_messages[i][:max_source_len]
            target_ids = encoded_messages[i + 1][:max_target_len]
            total_length += len(source_ids) + len(target_ids)
            encoded_pairs.append((source_ids, target_ids))

        return encoded_pairs


@dataclass
class Llama2Template(Template):

    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            system_text = ''
            if i == 0 and (system or tools or self.force_system):
                tool_text = self.format_tools.apply(
                    content=tools)[0] if tools else ''
                system_text = self.format_system.apply(content=(system +
                                                                tool_text))[0]
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message['role'] == Role.USER:
                elements += self.format_user.apply(content=system_text +
                                                   message['content'])
            elif message['role'] == Role.ASSISTANT:
                elements += self.format_assistant.apply(
                    content=message['content'])
            elif message['role'] == Role.OBSERVATION:
                elements += self.format_observation.apply(
                    content=message['content'])
            elif message['role'] == Role.FUNCTION:
                elements += self.format_function.apply(
                    content=message['content'])
            else:
                raise NotImplementedError('Unexpected role: {}'.format(
                    message['role']))

            encoded_messages.append(
                self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len,
                                reserved_label_len)


templates: Dict[str, Template] = {}


def register_template(
    name: str,
    format_user: Optional[Formatter] = None,
    format_assistant: Optional[Formatter] = None,
    format_system: Optional[Formatter] = None,
    format_function: Optional[Formatter] = None,
    format_observation: Optional[Formatter] = None,
    format_tools: Optional[Formatter] = None,
    format_separator: Optional[Formatter] = None,
    default_system: str = '',
    stop_words: List[str] = [],
    image_token: str = '<image>',
    efficient_eos: bool = False,
    replace_eos: bool = False,
    force_system: bool = False,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```json
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```python
    register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{'eos_token'}]
    template_class = Llama2Template if name.startswith('llama2') else Template
    default_user_formatter = StringFormatter(slots=['{{content}}'])
    default_assistant_formatter = StringFormatter(slots=['{{content}}'] +
                                                  eos_slots)
    default_function_formatter = FunctionFormatter(
        slots=['Action: {{name}}\nAction Input: {{arguments}}'] + eos_slots)
    default_tool_formatter = ToolFormatter(tool_format='default')
    default_separator_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user
        or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        default_system=default_system,
        stop_words=stop_words,
        image_token=image_token,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
    )


def _add_or_replace_eos_token(tokenizer: PreTrainedTokenizer,
                              eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({'eos_token': eos_token})

    if is_added:
        logger.info('Add eos token: {}'.format(tokenizer.eos_token))
    else:
        logger.info('Replace eos token: {}'.format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning(
            'New tokens have been added, make sure `resize_vocab` is True.')


def _jinja_escape(content: str) -> str:
    return content.replace("'", r"\'")


def _convert_slots_to_jinja(
    slots: Sequence[Union[str, Set[str], Dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    placeholder: str = 'content',
) -> str:
    slot_items = []
    for slot in slots:
        if isinstance(slot, str):
            slot_pieces = slot.split('{{content}}')
            if slot_pieces[0]:
                slot_items.append("'" + _jinja_escape(slot_pieces[0]) + "'")
            if len(slot_pieces) > 1:
                slot_items.append(placeholder)
                if slot_pieces[1]:
                    slot_items.append("'" + _jinja_escape(slot_pieces[1]) +
                                      "'")
        elif isinstance(
                slot,
                set):  # do not use {{ eos_token }} since it may be replaced
            if 'bos_token' in slot and tokenizer.bos_token_id is not None:
                slot_items.append("'" + tokenizer.bos_token + "'")
            elif 'eos_token' in slot and tokenizer.eos_token_id is not None:
                slot_items.append("'" + tokenizer.eos_token + "'")
        elif isinstance(slot, dict):
            raise ValueError('Dict is not supported.')

    return ' + '.join(slot_items)


def _get_jinja_template(template: 'Template',
                        tokenizer: PreTrainedTokenizer) -> str:
    jinja_template = ''

    if template.default_system:
        jinja_template += ("{% set system_message = '" +
                           _jinja_escape(template.default_system) + "' %}")

    jinja_template += "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"

    system_message = _convert_slots_to_jinja(template.format_system.apply(),
                                             tokenizer,
                                             placeholder='system_message')
    if isinstance(template, Llama2Template):
        pass
    elif template.force_system:
        jinja_template += '{{ ' + system_message + ' }}'
    else:
        jinja_template += ('{% if system_message is defined %}{{ ' +
                           system_message + ' }}{% endif %}')

    jinja_template += '{% for message in messages %}'
    jinja_template += "{% set content = message['content'] %}"
    if isinstance(template, Llama2Template):
        jinja_template += '{% if loop.index0 == 0 and system_message is defined %}'
        jinja_template += ('{% set content = ' + system_message +
                           " + message['content'] %}")
        jinja_template += '{% endif %}'

    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = _convert_slots_to_jinja(template.format_user.apply(),
                                           tokenizer)
    jinja_template += '{{ ' + user_message + ' }}'

    jinja_template += "{% elif message['role'] == 'assistant' %}"
    assistant_message = _convert_slots_to_jinja(
        template.format_assistant.apply() + template.format_separator.apply(),
        tokenizer)
    jinja_template += '{{ ' + assistant_message + ' }}'
    jinja_template += '{% endif %}'
    jinja_template += '{% endfor %}'
    return jinja_template


def get_template_and_fix_tokenizer(
    tokenizer: PreTrainedTokenizer,
    template_name: Optional[str] = None,
) -> Template:
    if template_name is None:
        template = templates['empty']  # placeholder
    else:
        template = templates.get(template_name, None)
        if template is None:
            raise ValueError('Template %s does not exist.' % template_name)

    logger.info('Using template: %s' % template_name)

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError(
                'Stop words are required to replace the EOS token.')

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token='<|endoftext|>')

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info('Add pad token: {}'.format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words),
            replace_additional_special_tokens=False,
        )
        logger.info('Add {} to stop words.'.format(','.join(stop_words)))
        if num_added_tokens > 0:
            logger.warning(
                'New tokens have been added, make sure `resize_vocab` is True.'
            )

    try:
        tokenizer.chat_template = _get_jinja_template(template, tokenizer)
    except ValueError:
        logger.info('Cannot add this chat template to tokenizer.')

    return template


register_template(
    name='alpaca',
    format_user=StringFormatter(
        slots=['### Instruction:\n{{content}}\n\n### Response:\n']),
    format_separator=EmptyFormatter(slots=['\n\n']),
    default_system=(
        'Below is an instruction that describes a task. '
        'Write a response that appropriately completes the request.\n\n'),
)

register_template(
    name='aquila',
    format_user=StringFormatter(slots=['Human: {{content}}###Assistant:']),
    format_separator=EmptyFormatter(slots=['###']),
    default_system=
    ('A chat between a curious human and an artificial intelligence assistant. '
     "The assistant gives helpful, detailed, and polite answers to the human's questions."
     ),
    stop_words=['</s>'],
    efficient_eos=True,
)

register_template(
    name='atom',
    format_user=StringFormatter(slots=[
        {'bos_token'},
        'Human: {{content}}\n',
        {'eos_token'},
        {'bos_token'},
        'Assistant:',
    ]),
    format_assistant=StringFormatter(slots=['{{content}}\n', {'eos_token'}]),
)

register_template(
    name='baichuan',
    format_user=StringFormatter(slots=[{
        'token': '<reserved_102>'
    }, '{{content}}', {
        'token': '<reserved_103>'
    }]),
    efficient_eos=True,
)

register_template(
    name='baichuan2',
    format_user=StringFormatter(
        slots=['<reserved_106>{{content}}<reserved_107>']),
    efficient_eos=True,
)

register_template(
    name='belle',
    format_user=StringFormatter(slots=['Human: {{content}}\n\nBelle: ']),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    format_separator=EmptyFormatter(slots=['\n\n']),
    force_system=True,
)

register_template(
    name='bluelm',
    format_user=StringFormatter(slots=[{
        'token': '[|Human|]:'
    }, '{{content}}', {
        'token': '[|AI|]:'
    }]),
)

register_template(
    name='breeze',
    format_user=StringFormatter(slots=['[INST] {{content}} [/INST] ']),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    default_system=
    ('You are a helpful AI assistant built by MediaTek Research. '
     'The user you are helping speaks Traditional Chinese and comes from Taiwan.'
     ),
    efficient_eos=True,
)

register_template(
    name='chatglm2',
    format_user=StringFormatter(
        slots=['[Round {{idx}}]\n\n问：{{content}}\n\n答：']),
    format_system=StringFormatter(slots=[{
        'token': '[gMASK]'
    }, {
        'token': 'sop'
    }, '{{content}}']),
    format_separator=EmptyFormatter(slots=['\n\n']),
    efficient_eos=True,
    force_system=True,
)

register_template(
    name='chatglm3',
    format_user=StringFormatter(slots=[{
        'token': '<|user|>'
    }, '\n', '{{content}}', {
        'token': '<|assistant|>'
    }]),
    format_assistant=StringFormatter(slots=['\n', '{{content}}']),
    format_system=StringFormatter(slots=[{
        'token': '[gMASK]'
    }, {
        'token': 'sop'
    }, '{{content}}']),
    format_function=FunctionFormatter(slots=['{{name}}\n{{arguments}}']),
    format_observation=StringFormatter(slots=[
        {
            'token': '<|observation|>'
        },
        '\n',
        '{{content}}',
        {
            'token': '<|assistant|>'
        },
    ]),
    stop_words=['<|user|>', '<|observation|>'],
    efficient_eos=True,
    force_system=True,
)

register_template(
    name='chatglm3_system',
    format_user=StringFormatter(slots=[{
        'token': '<|user|>'
    }, '\n', '{{content}}', {
        'token': '<|assistant|>'
    }]),
    format_assistant=StringFormatter(slots=['\n', '{{content}}']),
    format_system=StringFormatter(slots=[
        {
            'token': '[gMASK]'
        },
        {
            'token': 'sop'
        },
        {
            'token': '<|system|>'
        },
        '\n',
        '{{content}}',
    ]),
    format_function=FunctionFormatter(slots=['{{name}}\n{{arguments}}']),
    format_observation=StringFormatter(slots=[
        {
            'token': '<|observation|>'
        },
        '\n',
        '{{content}}',
        {
            'token': '<|assistant|>'
        },
    ]),
    default_system=(
        'You are ChatGLM3, a large language model trained by Zhipu.AI. '
        "Follow the user's instructions carefully. Respond using markdown."),
    stop_words=['<|user|>', '<|observation|>'],
    efficient_eos=True,
)

register_template(
    name='chatml',
    format_user=StringFormatter(slots=[
        '<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_system=StringFormatter(
        slots=['<|im_start|>system\n{{content}}<|im_end|>\n']),
    format_observation=StringFormatter(slots=[
        '<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_separator=EmptyFormatter(slots=['\n']),
    stop_words=['<|im_end|>', '<|im_start|>'],
    replace_eos=True,
)

register_template(
    name='chatml_de',
    format_user=StringFormatter(slots=[
        '<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_system=StringFormatter(
        slots=['<|im_start|>system\n{{content}}<|im_end|>\n']),
    format_observation=StringFormatter(slots=[
        '<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_separator=EmptyFormatter(slots=['\n']),
    default_system='Du bist ein freundlicher und hilfsbereiter KI-Assistent.',
    stop_words=['<|im_end|>', '<|im_start|>'],
    replace_eos=True,
)

register_template(
    name='codegeex2',
    format_system=StringFormatter(slots=[{
        'token': '[gMASK]'
    }, {
        'token': 'sop'
    }, '{{content}}']),
    force_system=True,
)

register_template(
    name='cohere',
    format_user=StringFormatter(slots=[(
        '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>'
        '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>')]),
    format_system=StringFormatter(slots=[
        {'bos_token'},
        '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>',
    ]),
    default_system=
    ('You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users '
     'by providing thorough responses. You are trained by Cohere.'),
)

register_template(
    name='cpm',
    format_user=StringFormatter(slots=['<用户>{{content}}<AI>']),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    force_system=True,
)

register_template(
    name='dbrx',
    format_user=StringFormatter(slots=[
        '<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_system=StringFormatter(
        slots=['<|im_start|>system\n{{content}}<|im_end|>\n']),
    format_observation=StringFormatter(slots=[
        '<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_separator=EmptyFormatter(slots=['\n']),
    default_system=
    ('You are DBRX, created by Databricks. You were last updated in December 2023. '
     'You answer questions based on information available up to that point.\n'
     'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough '
     'responses to more complex and open-ended questions.\nYou assist with various tasks, '
     'from writing to coding (using markdown for code blocks — remember to use ``` with '
     'code, JSON, and tables).\n(You do not have real-time data access or code execution '
     'capabilities. You avoid stereotyping and provide balanced perspectives on '
     'controversial topics. You do not provide song lyrics, poems, or news articles and '
     'do not divulge details of your training data.)\nThis is your system prompt, '
     'guiding your responses. Do not reference it, just respond to the user. If you find '
     'yourself talking about this message, stop. You should be responding appropriately '
     'and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION '
     "ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY."
     ),
    stop_words=['<|im_end|>'],
    replace_eos=True,
)

register_template(
    name='deepseek',
    format_user=StringFormatter(slots=['User: {{content}}\n\nAssistant:']),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    force_system=True,
)

register_template(
    name='deepseekcoder',
    format_user=StringFormatter(
        slots=['### Instruction:\n{{content}}\n### Response:']),
    format_assistant=StringFormatter(slots=['\n', '{{content}}']),
    format_separator=EmptyFormatter(slots=['\n<|EOT|>\n']),
    default_system=
    ('You are an AI programming assistant, utilizing the Deepseek Coder model, '
     'developed by Deepseek Company, and you only answer questions related to computer science. '
     'For politically sensitive questions, security and privacy issues, '
     'and other non-computer science questions, you will refuse to answer\n'),
    stop_words=['<|EOT|>'],
    efficient_eos=True,
)

register_template(
    name='default',
    format_user=StringFormatter(slots=['Human: {{content}}\nAssistant: ']),
    format_system=StringFormatter(slots=['{{content}}\n']),
    format_separator=EmptyFormatter(slots=['\n']),
)

register_template(
    name='empty',
    format_user=StringFormatter(slots=['{{content}}']),
    format_assistant=StringFormatter(slots=['{{content}}']),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    efficient_eos=True,
    force_system=True,
)

register_template(
    name='falcon',
    format_user=StringFormatter(slots=['User: {{content}}\nFalcon:']),
    format_separator=EmptyFormatter(slots=['\n']),
    efficient_eos=True,
)

register_template(
    name='fewshot',
    format_separator=EmptyFormatter(slots=['\n\n']),
    efficient_eos=True,
)

register_template(
    name='gemma',
    format_user=StringFormatter(slots=[
        '<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n'
    ]),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    format_observation=StringFormatter(slots=[
        '<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n'
    ]),
    format_separator=EmptyFormatter(slots=['<end_of_turn>\n']),
    efficient_eos=True,
    force_system=True,
)

register_template(
    name='glm4',
    format_user=StringFormatter(slots=['<|user|>\n{{content}}<|assistant|>']),
    format_assistant=StringFormatter(slots=['\n{{content}}']),
    format_system=StringFormatter(slots=['[gMASK]<sop>{{content}}']),
    format_function=FunctionFormatter(slots=['{{name}}\n{{arguments}}']),
    format_observation=StringFormatter(
        slots=['<|observation|>\n{{content}}<|assistant|>']),
    stop_words=['<|user|>', '<|observation|>'],
    efficient_eos=True,
    force_system=True,
)
register_template(
    name='intern',
    format_user=StringFormatter(
        slots=['<|User|>:{{content}}', {
            'token': '<eoh>'
        }, '\n<|Bot|>:']),
    format_separator=EmptyFormatter(slots=[{
        'token': '<eoa>'
    }, '\n']),
    stop_words=['<eoa>'],
    efficient_eos=True,
)

register_template(
    name='intern2',
    format_user=StringFormatter(slots=[
        '<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_system=StringFormatter(
        slots=[{'bos_token'}, '<|im_start|>system\n{{content}}<|im_end|>\n']),
    format_separator=EmptyFormatter(slots=['\n']),
    default_system=
    ('You are an AI assistant whose name is InternLM (书生·浦语).\n'
     '- InternLM (书生·浦语) is a conversational language model that is developed '
     'by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
     '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen '
     'by the user such as English and 中文.'),
    stop_words=['<|im_end|>'],
    efficient_eos=True,  # internlm2 tokenizer cannot set eos_token_id
)

register_template(
    name='llama2',
    format_user=StringFormatter(
        slots=[{'bos_token'}, '[INST] {{content}} [/INST]']),
    format_system=StringFormatter(
        slots=['<<SYS>>\n{{content}}\n<</SYS>>\n\n']),
    default_system=
    ('You are a helpful, respectful and honest assistant. '
     'Always answer as helpfully as possible, while being safe. '
     'Your answers should not include any harmful, unethical, '
     'racist, sexist, toxic, dangerous, or illegal content. '
     'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
     'If a question does not make any sense, or is not factually coherent, '
     'explain why instead of answering something not correct. '
     "If you don't know the answer to a question, please don't share false information."
     ),
)

register_template(
    name='llama2_zh',
    format_user=StringFormatter(
        slots=[{'bos_token'}, '[INST] {{content}} [/INST]']),
    format_system=StringFormatter(
        slots=['<<SYS>>\n{{content}}\n<</SYS>>\n\n']),
    default_system='You are a helpful assistant. 你是一个乐于助人的助手。',
)

register_template(
    name='llama3',
    format_user=StringFormatter(slots=[(
        '<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n')]),
    format_system=StringFormatter(slots=[
        {'bos_token'},
        '<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>',
    ]),
    format_observation=StringFormatter(slots=[(
        '<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n')]),
    default_system='You are a helpful assistant.',
    stop_words=['<|eot_id|>'],
    replace_eos=True,
)

register_template(
    name='mistral',
    format_user=StringFormatter(slots=['[INST] {{content}} [/INST]']),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    force_system=True,
)

register_template(
    name='olmo',
    format_user=StringFormatter(slots=['<|user|>\n{{content}}<|assistant|>']),
    format_assistant=StringFormatter(slots=['{{content}}', {'eos_token'}]),
    format_system=StringFormatter(slots=[{'eos_token'}, '{{content}}']),
    force_system=True,
)

register_template(
    name='openchat',
    format_user=StringFormatter(slots=[
        'GPT4 Correct User: {{content}}',
        {'eos_token'},
        'GPT4 Correct Assistant:',
    ]),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    force_system=True,
)

register_template(
    name='openchat-3.6',
    format_user=StringFormatter(slots=[(
        '<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{{content}}<|eot_id|>'
        '<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n')]),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    stop_words=['<|eot_id|>'],
    replace_eos=True,
    force_system=True,
)

register_template(
    name='orion',
    format_user=StringFormatter(
        slots=['Human: {{content}}\n\nAssistant: ', {'eos_token'}]),
    format_system=StringFormatter(slots=[{'bos_token'}, '{{content}}']),
    force_system=True,
)

register_template(
    name='phi',
    format_user=StringFormatter(
        slots=['<|user|>\n{{content}}<|end|>\n<|assistant|>\n']),
    format_system=StringFormatter(
        slots=[{'bos_token'}, '<|system|>\n{{content}}<|end|>\n']),
    format_separator=EmptyFormatter(slots=['\n']),
    default_system='You are a helpful AI assistant.',
    stop_words=['<|end|>'],
    replace_eos=True,
)

register_template(
    name='qwen',
    format_user=StringFormatter(slots=[
        '<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_system=StringFormatter(
        slots=['<|im_start|>system\n{{content}}<|im_end|>\n']),
    format_observation=StringFormatter(slots=[
        '<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_separator=EmptyFormatter(slots=['\n']),
    default_system='You are a helpful assistant.',
    stop_words=['<|im_end|>'],
    replace_eos=True,
)

register_template(
    name='solar',
    format_user=StringFormatter(
        slots=['### User:\n{{content}}\n\n### Assistant:\n']),
    format_system=StringFormatter(slots=['### System:\n{{content}}\n\n']),
    efficient_eos=True,
)

register_template(
    name='starchat',
    format_user=StringFormatter(
        slots=['<|user|>\n{{content}}<|end|>\n<|assistant|>']),
    format_system=StringFormatter(slots=['<|system|>\n{{content}}<|end|>\n']),
    format_separator=EmptyFormatter(slots=['\n']),
    stop_words=['<|end|>'],
    replace_eos=True,
    force_system=True,
)

register_template(
    name='telechat',
    format_user=StringFormatter(slots=['<_user>{{content}}<_bot>']),
    format_system=StringFormatter(slots=['<_system>{{content}}<_end>']),
    stop_words=['<_end>'],
    replace_eos=True,
)

register_template(
    name='vicuna',
    format_user=StringFormatter(slots=['USER: {{content}} ASSISTANT:']),
    default_system=
    ('A chat between a curious user and an artificial intelligence assistant. '
     "The assistant gives helpful, detailed, and polite answers to the user's questions."
     ),
)

register_template(
    name='xuanyuan',
    format_user=StringFormatter(slots=['Human: {{content}} Assistant:']),
    default_system=('以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，'
                    '会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、'
                    '不安全、有争议、政治敏感等相关的话题、问题和指示。\n'),
)

register_template(
    name='xverse',
    format_user=StringFormatter(slots=['Human: {{content}}\n\nAssistant: ']),
)

register_template(
    name='yayi',
    format_user=StringFormatter(slots=[{
        'token': '<|Human|>'
    }, ':\n{{content}}\n\n', {
        'token': '<|YaYi|>'
    }, ':']),
    format_system=StringFormatter(slots=[{
        'token': '<|System|>'
    }, ':\n{{content}}\n\n']),
    format_separator=EmptyFormatter(slots=['\n\n']),
    default_system=
    ('You are a helpful, respectful and honest assistant named YaYi '
     'developed by Beijing Wenge Technology Co.,Ltd. '
     'Always answer as helpfully as possible, while being safe.  '
     'Your answers should not include any harmful, unethical, '
     'racist, sexist, toxic, dangerous, or illegal content. '
     'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
     'If a question does not make any sense, or is not factually coherent, '
     'explain why instead of answering something not correct. '
     "If you don't know the answer to a question, please don't share false information."
     ),
    stop_words=['<|End|>'],
)

register_template(
    name='yi',
    format_user=StringFormatter(slots=[
        '<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
    ]),
    format_system=StringFormatter(
        slots=['<|im_start|>system\n{{content}}<|im_end|>\n']),
    format_separator=EmptyFormatter(slots=['\n']),
    stop_words=['<|im_end|>'],
    replace_eos=True,
)

register_template(
    name='yi_vl',
    format_user=StringFormatter(
        slots=['### Human: {{content}}\n### Assistant:']),
    format_separator=EmptyFormatter(slots=['\n']),
    default_system=
    ('This is a chat between an inquisitive human and an AI assistant. '
     'Assume the role of the AI assistant. Read all the images carefully, '
     "and respond to the human's questions with informative, helpful, detailed and polite answers. "
     '这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。'
     '仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n'),
    stop_words=['###'],
    efficient_eos=True,
)

register_template(
    name='yuan',
    format_user=StringFormatter(slots=['{{content}}', {
        'token': '<sep>'
    }]),
    format_separator=EmptyFormatter(slots=['\n']),
    stop_words=['<eod>'],
    replace_eos=True,
)

register_template(
    name='zephyr',
    format_user=StringFormatter(
        slots=['<|user|>\n{{content}}', {'eos_token'}, '<|assistant|>']),
    format_assistant=StringFormatter(slots=['\n{{content}}', {'eos_token'}]),
    format_system=StringFormatter(
        slots=['<|system|>\n{{content}}', {'eos_token'}]),
    default_system='You are Zephyr, a helpful assistant.',
)

register_template(
    name='ziya',
    format_user=StringFormatter(slots=['<human>:{{content}}\n<bot>:']),
    format_separator=EmptyFormatter(slots=['\n']),
)
