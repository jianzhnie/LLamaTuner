from threading import Thread

import torch
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)
import sys
sys.path.append("../")
from chatllms.configs import GenerationArguments, ModelInferenceArguments
from chatllms.utils.model_utils import get_logits_processor


def main(model_server_args, generation_args):
    """
    单轮对话，不具有对话历史的记忆功能
    """

    # Parse command-line arguments
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        model_server_args.model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto').to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_server_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    input_pattern = '<s>{}</s>'
    text = input('User：')
    while True:
        text = text.strip()
        text = input_pattern.format(text)
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
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
        print('Firefly：{}'.format(response))
        text = input('User：')


if __name__ == '__main__':
    parser = transformers.HfArgumentParser(
        (ModelInferenceArguments, GenerationArguments))
    model_server_args, generation_args = parser.parse_args_into_dataclasses()
    main(model_server_args, generation_args)
