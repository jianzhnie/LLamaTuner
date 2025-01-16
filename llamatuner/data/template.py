from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from transformers import PreTrainedTokenizer
from typing_extensions import override

from llamatuner.configs import DataArguments
from llamatuner.data.formatter import (EmptyFormatter, Formatter,
                                       FunctionFormatter, StringFormatter,
                                       ToolFormatter)
from llamatuner.data.tool_utils import FunctionCall
from llamatuner.data.utils import Role
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
    format_prefix: Formatter
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool

    def encode_oneturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
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

        Returns:

            返回一个元组，包含两个列表：提示（prompt_ids）和响应（answer_ids）的 token IDs。
            Tuple[List[int], List[int]]: Encoded prompt and response token IDs.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids
        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        """
        Returns multiple pairs of token IDs representing prompts and responses respectively.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to convert text to tokens.
            messages (List[Dict[str, str]]): List of message dictionaries containing roles and content.
            system (Optional[str]): System message to include at the beginning.
            tools (Optional[str]): Tools information to include.

        Returns:
            Sequence[Tuple[List[int], List[int]]]: Encoded prompt and response token ID pairs.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1])
                for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, List[FunctionCall]]:
        r"""
        Extracts tool message.
        """
        return self.format_tools.extract(content)

    def get_stop_token_ids(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        r"""
        Returns stop token ids.
        """
        stop_token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            stop_token_ids.add(tokenizer.convert_tokens_to_ids(token))

        return list(stop_token_ids)

    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> Sequence[Tuple[List[int], List[int]]]:
        """
        Encodes formatted inputs to pairs of token IDs.

        Turn 0: prefix + system + query        response
        Turn t: sep + query                    response

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to convert text to tokens.
            messages (List[Dict[str, str]]): List of message dictionaries containing roles and content.
            system (Optional[str]): System message to include at the beginning.
            tools (Optional[str]): Tools information to include.

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

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = (self.format_tools.apply(
                        content=tools)[0] if tools else '')
                    elements += self.format_system.apply(content=(system +
                                                                  tool_text))

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
                    f"Unexpected role: {message['role']}")

            encoded_messages.append(
                self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

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


@dataclass
class Llama2Template(Template):

    @override
    def _encode(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
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
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = (self.format_tools.apply(
                        content=tools)[0] if tools else '')
                    system_text = self.format_system.apply(
                        content=(system + tool_text))[0]

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

        return encoded_messages


templates: Dict[str, Template] = {}


def register_template(
    name: str,
    format_user: Optional[Formatter] = None,
    format_assistant: Optional[Formatter] = None,
    format_system: Optional[Formatter] = None,
    format_function: Optional[Formatter] = None,
    format_observation: Optional[Formatter] = None,
    format_tools: Optional[Formatter] = None,
    format_prefix: Optional[Formatter] = None,
    default_system: str = '',
    stop_words: Optional[Sequence[str]] = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    replace_jinja_template: bool = False,
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
    template_class = (Llama2Template if any(
        k in name for k in ('llama2', 'mistral', 'pixtral')) else Template)
    default_slots = ['{{content}}'
                     ] if efficient_eos else ['{{content}}', {'eos_token'}]
    default_user_formatter = StringFormatter(slots=['{{content}}'])
    default_assistant_formatter = StringFormatter(slots=default_slots)
    default_function_formatter = FunctionFormatter(slots=default_slots,
                                                   tool_format='default')
    default_tool_formatter = ToolFormatter(tool_format='default')
    default_prefix_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user
        or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
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
        elif isinstance(slot, set):
            if 'bos_token' in slot and tokenizer.bos_token_id is not None:
                slot_items.append("'" + tokenizer.bos_token + "'")
            elif 'eos_token' in slot and tokenizer.eos_token_id is not None:
                slot_items.append("'" + tokenizer.eos_token + "'")
        elif isinstance(slot, dict):
            raise ValueError('Dict is not supported.')

    return ' + '.join(slot_items)


def _get_jinja_template(template: Template,
                        tokenizer: PreTrainedTokenizer) -> str:
    jinja_template = ''

    prefix = _convert_slots_to_jinja(template.format_prefix.apply(), tokenizer)
    if prefix:
        jinja_template += '{{ ' + prefix + ' }}'

    if template.default_system:
        jinja_template += ("{% set system_message = '" +
                           _jinja_escape(template.default_system) + "' %}")

    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
    )
    system_message = _convert_slots_to_jinja(template.format_system.apply(),
                                             tokenizer,
                                             placeholder='system_message')

    if isinstance(template, Llama2Template):
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
        template.format_assistant.apply(), tokenizer)
    jinja_template += '{{ ' + assistant_message + ' }}'
    jinja_template += '{% endif %}'
    jinja_template += '{% endfor %}'
    return jinja_template


def get_template_and_fix_tokenizer(tokenizer: PreTrainedTokenizer,
                                   data_args: DataArguments) -> Template:
    if data_args.template is None:
        template = templates['empty']  # placeholder
    else:
        template = templates.get(data_args.template, None)
        if template is None:
            raise ValueError(f'Template {data_args.template} does not exist.')

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError(
            'Current template does not support `train_on_prompt`.')

    if data_args.tool_format is not None:
        logger.info(f'Using tool format: {data_args.tool_format}.')
        default_slots = (['{{content}}'] if template.efficient_eos else
                         ['{{content}}', {'eos_token'}])
        template.format_function = FunctionFormatter(
            slots=default_slots, tool_format=data_args.tool_format)
        template.format_tools = ToolFormatter(
            tool_format=data_args.tool_format)

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
    name='empty',
    efficient_eos=True,
)
