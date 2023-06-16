import argparse
from typing import Union

import gradio as gr
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils.apply_lora import apply_lora
from utils.callbacks import Iteratorize, Stream


class Prompter(object):
    def __init__(self) -> None:
        self.PROMPT_DICT = {
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
        self.reponse_split = '### Response:'

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        response: Union[None, str] = None,
    ):
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
        return output.split(self.reponse_split)[1].strip()


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to pre-trained model')
    parser.add_argument('--lora_model_name_or_path',
                        default=None,
                        type=str,
                        help='Path to pre-trained model')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='Avoid using CUDA when available')
    parser.add_argument('--load_8bit',
                        action='store_true',
                        help='Whether to use load_8bit  instead of 32-bit')
    args = parser.parse_args()

    args.device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    return args


def main(args):
    if args.lora_model_name_or_path is not None:
        model, tokenizer = apply_lora(args.model_name_or_path,
                                      args.lora_model_name_or_path,
                                      load_8bit=args.load_8bit)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True)

    if not args.load_8bit:
        model.half()  # seems to fix bugs for some users.
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    prompter = Prompter()

    def evaluate(
        instruction,
        input=None,
        temperature=0.8,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(args.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,
            no_repeat_ngram_size=6,
            repetition_penalty=1.8,
            **kwargs,
        )

        generate_params = {
            'input_ids': input_ids,
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
                input_ids=input_ids,
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
                                 value=0.1,
                                 label='Temperature'),
            gr.components.Slider(minimum=0,
                                 maximum=1,
                                 value=0.75,
                                 label='Top p'),
            gr.components.Slider(minimum=0,
                                 maximum=100,
                                 step=1,
                                 value=40,
                                 label='Top k'),
            gr.components.Slider(minimum=1,
                                 maximum=4,
                                 step=1,
                                 value=4,
                                 label='Beams'),
            gr.components.Slider(minimum=1,
                                 maximum=2000,
                                 step=1,
                                 value=128,
                                 label='Max tokens'),
            gr.components.Checkbox(label='Stream output'),
        ],
        outputs=[gr.inputs.Textbox(
            lines=5,
            label='Output',
        )],
        title='Baichuan7B',
        description=description,
    )

    server.queue().launch(server_name='0.0.0.0', share=False)


if __name__ == '__main__':
    args = args_parser()
    main(args)
