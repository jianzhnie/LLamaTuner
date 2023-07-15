import logging
import os
import platform
from threading import Thread
from typing import List, Tuple

import transformers
from transformers import (PreTrainedModel, PreTrainedTokenizer,
                          TextIteratorStreamer)

from chatllms.model.load_pretrain_model import load_model_tokenizer
from chatllms.utils.config import GenerationArguments, ModelInferenceArguments
from chatllms.utils.model_utils import get_logits_processor
from chatllms.utils.template import PromptTemplate

logger = logging.getLogger(__name__)


def predict_and_print(
    query: str,
    history: List[Tuple[str, str]],
    source_prefix: str,
    prompt_template: PromptTemplate,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    generation_args: dict,
) -> List[str]:
    """
    Generates a response to the given query using GPT-3.5 model and prints it to the console.

    Args:
        query (str): The input query for which a response is to be generated.
        history (List[Tuple[str, str]]): A list of previous queries and their responses.
        source_prefix (str): The prefix string added to the beginning of each input sequence.
        prompt_template (PromptTemplate): The prompt template used to generate the input sequence to the model.
        tokenizer (PreTrainedTokenizer): The tokenizer used to convert the raw text into input tokens.
        model (PreTrainedModel): The GPT-3.5 model used to generate the response.
        generation_args (dict): A dictionary containing the arguments to be passed to the generate() method of the model.

    Returns:
        List[Tuple[str, str]]: A list of all the previous queries and their responses, including the current one.
    """

    # Convert the query and history into input IDs
    input_text = prompt_template.get_prompt(query, history, source_prefix)
    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Create a TextIteratorStreamer object to stream the response from the model
    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=60.0,
                                    skip_prompt=True,
                                    skip_special_tokens=True)

    # Set the arguments for the model's generate() method
    gen_kwargs = dict(
        inputs,
        streamer=streamer,
        logits_processor=get_logits_processor(),
        **generation_args.to_dict(),
    )

    # Start a separate thread to generate the response asynchronously
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Print the model name and the response as it is generated
    print('Assistant: ', end='', flush=True)
    response = ''
    for new_text in streamer:
        print(new_text, end='', flush=True)
        response += new_text
    print()

    # Update the history with the current query and response and return it
    history.append((query, response))
    return history


def main():
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'

    # Parse command-line arguments
    parser = transformers.HfArgumentParser(
        (ModelInferenceArguments, GenerationArguments))
    model_server_args, generation_args = parser.parse_args_into_dataclasses()
    # Load the model and tokenizer
    model, tokenizer = load_model_tokenizer(
        args=model_server_args,
        checkpoint_dir=model_server_args.checkpoint_dir,
        is_trainable=False,
        logger=logger,
    )

    prompt_template = PromptTemplate(model_server_args.prompt_template)
    source_prefix = model_server_args.source_prefix if model_server_args.source_prefix else ''
    model_name = 'BLOOM' if 'bloom' in model_server_args.model_name_or_path else 'LLaMA'
    history: List[str] = []
    print(f'欢迎使用 {model_name} 模型，输入内容即可对话，clear 清空对话历史，stop 终止程序')
    while True:
        try:
            query = input('\nInput: ')
        except UnicodeDecodeError:
            print(
                'Detected decoding error at the inputs, please set the terminal encoding to utf-8.'
            )
            continue
        if query.strip() == 'stop':
            break

        if query.strip() == 'clear':
            # Clear the conversation history
            history = []
            os.system(clear_command)
            print('History has been removed.')
            print(f'欢迎使用 {model_name} 模型，输入内容即可对话，clear 清空对话历史，stop 终止程序')
            continue

        # Perform prediction and printing
        history = predict_and_print(query, history, source_prefix,
                                    prompt_template, tokenizer, model,
                                    generation_args)


if __name__ == '__main__':
    main()
