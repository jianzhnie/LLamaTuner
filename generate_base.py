import argparse
import sys
from typing import Union

import gradio as gr
import torch
sys.path.append('../../')
from transformers import GenerationConfig

from utils.apply_lora import apply_lora
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def evaluate(
        input=None,
        temperature=0.8,
        top_p=0.75,
        top_k=40,
        max_new_tokens=128,
        **kwargs,
    ):
        inputs = tokenizer(input, return_tensors='pt')
        inputs = inputs.to(args.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            no_repeat_ngram_size=6,
            repetition_penalty=1.8,
            **kwargs,
        )
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
        output = tokenizer.decode(s, skip_special_tokens=True)
        yield output


    description = 'Baichuan7B is a 7B-parameter LLaMA model finetuned to follow instructions.'
    server = gr.Interface(
        fn=evaluate,
        inputs=[
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
                                 maximum=2000,
                                 step=1,
                                 value=128,
                                 label='Max tokens'),
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
