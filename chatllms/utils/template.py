import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate(object):
    """
    A template for formatting a conversation prompt.

    Args:
        name: Name of template
        prefix: Prefix text
        prompt: Prompt text
        sep: Separator between prompts
        use_history: Whether to use conversation history

    """

    name: str
    prefix: str = ''
    prompt: str = None
    sep: str = None
    use_history: bool = False

    def get_prompt(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = None,
    ) -> str:
        """
        Returns a string containing prompt without response.

        Args:
            query (str): The input query text.
            history (Optional[list], optional): The conversation history. Defaults to None.
            prefix (Optional[str], optional): The prefix text for the prompt. Defaults to ''.

        Returns:
            str: A string containing prompt without response.
        """
        return ''.join(self.format_example(query, history, prefix))

    def get_dialog(self,
                   query: str,
                   response: str,
                   history: Optional[List[Tuple[str, str]]] = None,
                   prefix: Optional[str] = None) -> List[str]:
        """
        Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.

        Args:
            query (str): The input query text.
            response (str): The response text.
            history (Optional[list], optional): The conversation history. Defaults to None.
            prefix (Optional[str], optional): The prefix text for the prompt. Defaults to ''.

        Returns:
            List[str]: A list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
        """
        return self.format_example(query, history, prefix) + [response]

    def format_example(self,
                       query: str,
                       history: Optional[List[Tuple[str, str]]] = None,
                       prefix: Optional[str] = None) -> List[str]:
        """
        Formats the conversation example.

        Args:
            query (str): The input query text.
            history (Optional[list], optional): The conversation history. Defaults to None.
            prefix (Optional[str], optional): The prefix text for the prompt. Defaults to ''.

        Returns:
            List[str]: A list containing formatted conversation examples.
        """
        prefix = prefix if prefix else self.prefix  # use prefix if provided
        prefix = prefix + self.sep if prefix else ''  # add separator for non-empty prefix
        history = history if (history and self.use_history) else []
        history = history + [(query, '<dummy>')]
        convs = []
        for turn_idx, (user_query, bot_resp) in enumerate(history):
            if turn_idx == 0:
                convs.append(prefix + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs[:-1]  # drop last

    def register_template(
        self,
        name: str,
        prefix: str,
        prompt: str,
        sep: str,
        use_history: Optional[bool] = True,
    ) -> None:
        """
        Registers a new conversation template.

        Args:
            prefix (str): The prefix text for the prompt.
            prompt (str): The prompt text.
            sep (str): The separator between different prompts.
            use_history (Optional[bool], optional): Whether to include conversation history. Defaults to True.
        """
        self.name = name
        self.prefix = prefix
        self.prompt = prompt
        self.sep = sep
        self.use_history = use_history

    def __post_init__(self):
        """
        Initializes the instance of the class.
        """
        if self.name == 'default':
            """
            Supports language model inference without histories.
            """
            self.register_template(name='vanilla',
                                   prefix='',
                                   prompt='<s>{query}</s>',
                                   sep='',
                                   use_history=False)

        elif self.name == 'llama2':
            r"""
            Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
                    https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
                    https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
            """
            self.register_template(
                name='llama2',
                prefix=
                '<<SYS>>\nYou are a helpful, respectful and honest assistant. '
                'Always answer as helpfully as possible, while being safe.  '
                'Your answers should not include any harmful, unethical, '
                'racist, sexist, toxic, dangerous, or illegal content. '
                'Please ensure that your responses are socially unbiased and positive in nature.\n'
                'If a question does not make any sense, or is not factually coherent, '
                'explain why instead of answering something not correct. '
                "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
                prompt=' [INST] {query} [/INST] ',
                sep='</s>',
                use_history=True)

        elif self.name == 'alpaca':
            """
            Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
                      https://github.com/ymcui/Chinese-LLaMA-Alpaca
            """
            self.register_template(
                name='alpaca',
                prefix='Below is an instruction that describes a task. '
                'Write a response that appropriately completes the request.',
                prompt='### Instruction:\n{query}\n\n### Response:\n',
                sep='\n\n',
                use_history=True)

        elif self.name == 'vicuna':
            """
            Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
                      https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
            """
            self.register_template(
                name='vicuna',
                prefix=
                'A chat between a curious user and an artificial intelligence assistant. '
                'The assistant gives helpful, detailed, and polite answers to the user\'s questions.',
                prompt='USER: {query} ASSISTANT: ',
                sep='</s>',
                use_history=True)

        elif self.name == 'belle':
            """
            Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
            """
            self.register_template(name='belle',
                                   prefix='',
                                   prompt='Human: {query}\n\nBelle: ',
                                   sep='\n\n',
                                   use_history=True)

        elif self.name == 'linly':
            """
            Supports: https://github.com/CVI-SZU/Linly
            """
            self.register_template(name='linly',
                                   prefix='',
                                   prompt='User: {query}\nBot: ',
                                   sep='\n',
                                   use_history=True)

        elif self.name == 'billa':
            """
            Supports: https://github.com/Neutralzz/BiLLa
            """
            self.register_template(name='billa',
                                   prefix='',
                                   prompt='Human: {query}\nAssistant: ',
                                   sep='\n',
                                   use_history=True)

        elif self.name == 'ziya':
            """
            Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
            """
            self.register_template(name='ziya',
                                   prefix='',
                                   prompt='<human>:{query}\n<bot>:',
                                   sep='\n',
                                   use_history=True)

        elif self.name == 'aquila':
            """
            Supports: https://huggingface.co/qhduan/aquilachat-7b
            """
            self.register_template(
                name='aquila',
                prefix=
                'A chat between a curious human and an artificial intelligence assistant. '
                'The assistant gives helpful, detailed, and polite answers to the human\'s questions.',
                prompt='Human: {query}###Assistant: ',
                sep='###',
                use_history=True)

        elif self.name == 'intern':
            r"""
            Supports: https://huggingface.co/internlm/internlm-chat-7b
            """
            self.register_template(name='intern',
                                   prefix='',
                                   prompt='<|User|>:{query}<eoh>\n<|Bot|>:',
                                   sep='<eoa>\n',
                                   use_history=True)

        elif self.name == 'baichuan':
            r"""
            Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
            """
            self.register_template(
                name='baichuan',
                prefix='',
                prompt='<reserved_102>{query}<reserved_103>',
                sep='</s>',
                use_history=True)

        else:
            raise NotImplementedError(f'Template {self.name} does not exist.')
