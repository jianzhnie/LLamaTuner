import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Dict, List, Literal, Optional, Sequence, Set, Tuple,
                    Union)

SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]

JSON_FORMAT_PROMPT = """, in a JSON format representing the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)"""

TOOL_SYSTEM_PROMPT = ('You have access to the following tools:\n{tool_text}'
                      'Use the following format if using a tool:\n'
                      '```\n'
                      'Action: tool name (one of [{tool_names}]).\n'
                      'Action Input: the input to the tool{format_prompt}.\n'
                      '```\n')


def default_tool_formatter(tools: List[Dict[str, Any]]) -> str:
    """
    Formats tool information into a human-readable string.

    Args:
        tools (List[Dict[str, Any]]): List of tool dictionaries.

    Returns:
        str: Formatted tool information.
    """
    tool_text = ''
    tool_names = []

    for tool in tools:
        param_text = ''
        for name, param in tool['parameters']['properties'].items():
            # Format each parameter's details
            required = (', required' if name in tool['parameters'].get(
                'required', []) else '')
            enum = (', should be one of [{}]'.format(', '.join(param['enum']))
                    if param.get('enum') else '')
            items = (', where each item should be {}'.format(
                param['items'].get('type', '')) if param.get('items') else '')
            param_text += f"  - {name} ({param.get('type', '')}{required}): {param.get('description', '')}{enum}{items}\n"

        # Format each tool's details
        tool_text += f"> Tool Name: {tool['name']}\nTool Description: {tool.get('description', '')}\nTool Args:\n{param_text}\n"
        tool_names.append(tool['name'])

    # Combine all formatted tool information into the final prompt
    return TOOL_SYSTEM_PROMPT.format(
        tool_text=tool_text,
        tool_names=', '.join(tool_names),
        format_prompt=JSON_FORMAT_PROMPT,
    )


def default_tool_extractor(content: str) -> Union[str, Tuple[str, str]]:
    """
    Extracts tool name and input arguments from the provided content.

    Args:
        content (str): The content to extract information from.

    Returns:
        Union[str, Tuple[str, str]]: Extracted tool name and arguments in JSON format, or original content if extraction fails.
    """
    regex = re.compile(r'Action:\s*([a-zA-Z0-9_]+).*?Action Input:\s*(.*)',
                       re.DOTALL)
    action_match = re.search(regex, content)

    if not action_match:
        return content

    tool_name = action_match.group(1).strip()
    tool_input = action_match.group(2).strip().strip('"').strip('```')

    try:
        arguments = json.loads(tool_input)
    except json.JSONDecodeError:
        return content

    return tool_name, json.dumps(arguments, ensure_ascii=False)


@dataclass
class Formatter(ABC):
    """
    Abstract base class for formatters. Defines the structure for formatters
    that manipulate sequences of strings, sets, or dictionaries based on specific rules.
    Formatter类是一个抽象基类，定义了所有格式化器必须实现的接口。

    Attributes:
        slots (Sequence[Union[str, Set[str], Dict[str, str]]]): The slots to format.
        tool_format (Optional[Literal["default"]]): Optional tool format specification.

        slots: 一个序列，包含字符串、集合或字典，这些元素将被格式化。
        tool_format: 可选的工具格式，可以设置为 "default"。
        apply: 一个抽象方法，要求子类实现具体的格式化逻辑。
        extract: 一个未实现的方法，子类可以根据需要重载这个方法来提取内容
    """

    slots: Sequence[Union[str, Set[str],
                          Dict[str, str]]] = field(default_factory=list)
    tool_format: Optional[Literal['default']] = None

    @abstractmethod
    def apply(self,
              **kwargs) -> Sequence[Union[str, Set[str], Dict[str, str]]]:
        """
        Applies formatting to the slots based on provided keyword arguments.

        Returns:
            Sequence[Union[str, Set[str], Dict[str, str]]]: The formatted slots.
        """
        ...

    def extract(self, content: str) -> Union[str, Tuple[str, str]]:
        """
        Extracts information from the provided content string. Must be implemented by subclasses.

        Args:
            content (str): The content to extract information from.

        Returns:
            Union[str, Tuple[str, str]]: Extracted information.
        """
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    """
    Formatter that ensures no placeholders are present in the slots.
    EmptyFormatter类 确保插槽（slots）中没有任何占位符
    """

    def __post_init__(self):
        """
        __post_init__方法：在类的初始化之后自动调用，\
            用于检查slots中是否包含任何占位符（如{{placeholder}}）。
            如果包含占位符，则抛出错误，因为空格式化器不应该包含占位符。

        Raises:
            ValueError: _description_
        """
        # Ensure no placeholders are present in the slots
        has_placeholder = any(
            isinstance(slot, str)
            and re.search(r'\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}', slot)
            for slot in self.slots)
        if has_placeholder:
            raise ValueError(
                'Empty formatter should not contain any placeholder.')

    def apply(self,
              **kwargs) -> Sequence[Union[str, Set[str], Dict[str, str]]]:
        """
        Returns the slots without any modification.

        Returns:
            Sequence[Union[str, Set[str], Dict[str, str]]]: The original slots.
        """
        return self.slots


class StringFormatter(Formatter):
    """
    Formatter that replaces placeholders in the slots with provided values.
    StringFormatter类 用于替换插槽中的占位符。
    """

    def __post_init__(self):
        # Ensure at least one placeholder is present in the slots
        has_placeholder = any(
            isinstance(slot, str)
            and re.search(r'\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}', slot)
            for slot in self.slots)
        if not has_placeholder:
            raise ValueError(
                'A placeholder is required in the string formatter.')

    def apply(self,
              **kwargs) -> Sequence[Union[str, Set[str], Dict[str, str]]]:
        """
        Replaces placeholders in the slots with provided values.

        Args:
            **kwargs: The values to replace the placeholders with.

        Returns:
            Sequence[Union[str, Set[str], Dict[str, str]]]: The formatted slots.

        Raises:
            RuntimeError: If a non-string value is provided for a placeholder.
        """
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError(
                            f'Expected a string, got {type(value)}')
                    slot = slot.replace(f'{{{{{name}}}}}', value)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(
                    f'Input must be string, set[str], or dict[str, str], got {type(slot)}'
                )
        return elements


@dataclass
class FunctionFormatter(Formatter):
    """
    Formatter that replaces placeholders for function name and arguments in the slots.
    FunctionFormatter 类 用于替换插槽中的函数名和参数占位符。
    """

    def __post_init__(self):
        # Ensure both name and arguments placeholders are present in the slots
        has_name = any(
            isinstance(slot, str) and '{{name}}' in slot
            for slot in self.slots)
        has_args = any(
            isinstance(slot, str) and '{{arguments}}' in slot
            for slot in self.slots)
        if not has_name or not has_args:
            raise ValueError(
                'Name and arguments placeholders are required in the function formatter.'
            )

    def apply(self,
              **kwargs) -> Sequence[Union[str, Set[str], Dict[str, str]]]:
        """
        Replaces placeholders for function name and arguments in the slots.

        Args:
            **kwargs: The function content in JSON format to extract name and arguments from.

        Returns:
            Sequence[Union[str, Set[str], Dict[str, str]]]: The formatted slots.

        Raises:
            RuntimeError: If the input slot is not a string, set, or dictionary.
        """
        content = kwargs.pop('content', '')
        try:
            function = json.loads(content)
            name = function.get('name', '')
            arguments = json.dumps(function.get('arguments', {}),
                                   ensure_ascii=False)
        except json.JSONDecodeError:
            name, arguments = '', ''

        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                slot = slot.replace('{{name}}',
                                    name).replace('{{arguments}}', arguments)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(
                    f'Input must be string, set[str], or dict[str, str], got {type(slot)}'
                )
        return elements


@dataclass
class ToolFormatter(Formatter):
    """ToolFormatter，用于处理工具格式的内容。"""

    def __post_init__(self):
        """
        Post-initialization check to ensure tool_format is specified.
        """
        if self.tool_format is None:
            raise ValueError('Tool format was not found.')

    def apply(self,
              **kwargs) -> Sequence[Union[str, Set[str], Dict[str, str]]]:
        """
        Apply the tool formatter to the provided content.

        apply 方法用于将输入的 content 格式化。具体步骤如下：

        - 从 kwargs 中提取 content。
        - 尝试将 content 解析为 JSON 对象。
        - 如果 tools 列表为空，则返回包含一个空字符串的列表。
        - 如果 tool_format 为 'default'，则调用 default_tool_formatter(tools) 函数进行格式化，并返回结果。
        - 如果 tool_format 不是 'default'，则抛出 NotImplementedError，表示尚未实现其他格式。
        - 如果解析 content 失败，则返回包含一个空字符串的列表。

        Args:
            **kwargs: Arbitrary keyword arguments, expected to include 'content'.

        Returns:
            Sequence[Union[str, Set[str], Dict[str, str]]]: Formatted tools information.
        """
        content = kwargs.pop('content', '')
        try:
            tools = json.loads(content)
            if not tools:
                return ['']

            if self.tool_format == 'default':
                return [default_tool_formatter(tools)]
            else:
                raise NotImplementedError(
                    f"Tool format '{self.tool_format}' is not implemented.")
        except json.JSONDecodeError:
            return ['']

    def extract(self, content: str) -> Union[str, Tuple[str, str]]:
        """
        Extract tool information from the content.

        Args:
            content (str): The content to extract information from.

        Returns:
            Union[str, Tuple[str, str]]: Extracted tool information.
        """
        if self.tool_format == 'default':
            return default_tool_extractor(content)
        else:
            raise NotImplementedError(
                f"Tool format '{self.tool_format}' is not implemented.")
