import os
from os.path import join
from typing import Dict

import bitsandbytes as bnb
import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict,
                                         tokenizer: PreTrainedTokenizer,
                                         model: PreTrainedModel):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (
        bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(f'trainable params: {trainable_params} || '
          f'all params: {all_param} || '
          f'trainable: {100 * trainable_params / all_param}')


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint,
                                             'adapter_model')
        else:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f'{PREFIX_CHECKPOINT_DIR}-{state.global_step}')

        peft_model_path = os.path.join(checkpoint_folder, 'adapter_model')
        kwargs['model'].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder,
                                          'pytorch_model.bin')
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)
