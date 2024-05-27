import argparse
import logging
from typing import Union

import gradio as gr
import torch
import transformers
from transformers import GenerationConfig

from chatllms.configs import ModelInferenceArguments
from chatllms.model.load_pretrain_model import load_model_tokenizer
from chatllms.utils.stream_server import Iteratorize, Stream

ALPACA_PROMPT_DICT = {
    'prompt_input':
    ('Below is an instruction that describes a task, paired with an input that provides further context. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
     ),
    'prompt_no_input':
    ('Below is an instruction that describes a task. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n### Response:'),
}

PROMPT_DICT = {
    'prompt_input': ('{instruction}\n\n### Response:'),
    'prompt_no_input': ('{instruction}\n\n### Response:'),
}

logger = logging.getLogger(__name__)


class Prompter:
    """A class for generating prompts and extracting responses from generated
    text."""

    def __init__(self, prompt_template: str = None):
        """Initializes a new instance of the Prompter class.

        Args:
            prompt_template (str): The name of the prompt template to use. Default is None.
                                   If set to 'alpaca', it will use a different set of prompt templates.
        """
        self.PROMPT_DICT = ALPACA_PROMPT_DICT if prompt_template == 'alpaca' else PROMPT_DICT
        self.reponse_split = '### Response:'

    def generate_prompt(self,
                        instruction: str,
                        input: Union[str, None] = None,
                        response: Union[str, None] = None) -> str:
        """Generates a prompt based on the specified inputs.

        Args:
            instruction (str): The instruction to include in the prompt.
            input (Union[str, None]): The input to include in the prompt. Default is None.
            response (Union[str, None]): The response to include in the prompt. Default is None.

        Returns:
            str: The generated prompt text.
        """
        prompt_input, prompt_no_input = self.PROMPT_DICT[
            'prompt_input'], self.PROMPT_DICT['prompt_no_input']

        if input is not None:
            prompt_text = prompt_input.format(instruction=instruction,
                                              input=input)
        else:
            prompt_text = prompt_no_input.format(instruction=instruction)

        if response:
            prompt_text = f'{prompt_text}{response}'

        return prompt_text

    def get_response(self, output: str) -> str:
        """Extracts the response from the generated text.

        Args:
            output (str): The generated text to extract the response from.

        Returns:
            str: The extracted response.
        """
        return output.split(self.reponse_split)[1].strip()


def main():
    parser = transformers.HfArgumentParser(ModelInferenceArguments)
    model_server_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    args = argparse.Namespace(**vars(model_server_args))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model_tokenizer(args,
                                            checkpoint_dir=args.checkpoint_dir,
                                            is_trainable=False,
                                            logger=logger)
    prompter = Prompter()

    def evaluate(
        instruction,
        input=None,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to(args.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,
            # no_repeat_ngram_size=6,
            # repetition_penalty=1.8,
            **kwargs,
        )

        generate_params = {
            'input_ids': inputs['input_ids'],
            'generation_config': generation_config,
            'return_dict_in_generate': True,
            'output_scores': True,
            'max_new_tokens': max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault('stopping_criteria',
                                  transformers.StoppingCriteriaList())
                kwargs['stopping_criteria'].append(
                    Stream(callback_func=callback))
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback,
                                   kwargs,
                                   callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    description = 'Baichuan7B is a 7B-parameter LLaMA model finetuned to follow instructions.'
    server = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(lines=2,
                                  label='Instruction',
                                  placeholder='Tell me about alpacas.'),
            gr.components.Textbox(lines=2, label='Input', placeholder='none'),
            gr.components.Slider(minimum=0,
                                 maximum=1,
                                 value=1.0,
                                 label='Temperature'),
            gr.components.Slider(minimum=0,
                                 maximum=1,
                                 value=1.0,
                                 label='Top p'),
            gr.components.Slider(minimum=0,
                                 maximum=100,
                                 step=1,
                                 value=50,
                                 label='Top k'),
            gr.components.Slider(minimum=1,
                                 maximum=4,
                                 step=1,
                                 value=4,
                                 label='Beams'),
            gr.components.Slider(minimum=16,
                                 maximum=1024,
                                 step=32,
                                 value=128,
                                 label='Max new tokens'),
            gr.components.Checkbox(label='Stream output'),
        ],
        outputs=[gr.inputs.Textbox(
            lines=5,
            label='Output',
        )],
        title='Baichuan7B',
        description=description,
    )

    server.queue().launch(server_name='0.0.0.0', share=True)


if __name__ == '__main__':
    main()
