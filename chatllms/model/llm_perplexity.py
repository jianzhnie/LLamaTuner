import logging
import os
import sys
from typing import List, Union

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser)

# Add parent directory to sys.path
sys.path.append('../../')

from chatllms.configs import ModelInferenceArguments
from chatllms.data.data_utils import IGNORE_INDEX
from chatllms.utils.model_utils import add_special_tokens_if_missing


class LLMPerplexity:
    """Language model to compute perplexity.

    Args:
        cache_dir (str): Directory to cache models.
        model_name_or_path (str): Model name or path to load from Hub.
        use_auth_token (bool): Whether to use auth token for loading model.
        trust_remote_code (bool): Whether to trust remote code.
        low_cpu_mem_usage (bool): Whether to use low CPU memory usage.
        max_length (int, optional): Max sequence length. Defaults to None.
        fp16 (bool): Whether to use 16-bit precision.
        device (str): Device to load model to.
    """

    def __init__(
        self,
        cache_dir: str = None,
        model_name_or_path: str = 'facebook/opt-125m',
        use_auth_token: bool = False,
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = False,
        max_length: int = None,
        fp16: bool = False,
        device: str = 'cpu',
    ):
        # Determine the torch data type based on the input arguments
        torch_dtype = torch.float16 if fp16 else torch.float32

        config_kwargs = {
            'cache_dir': cache_dir,
            'use_auth_token': use_auth_token,
            'trust_remote_code': trust_remote_code,
        }
        device_map = 'auto'

        # Set device map if running in distributed training (using environment variable LOCAL_RANK)
        if os.environ.get('LOCAL_RANK') is not None:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            device_map = {'': local_rank}

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side='right',
            use_fast=False,
            **config_kwargs,
        )
        self.config = AutoConfig.from_pretrained(model_name_or_path,
                                                 **config_kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=self.config,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **config_kwargs,
        ).to(device).eval()

        # Loss function
        self.loss_fct = CrossEntropyLoss(reduction='none')
        # Max length
        self.max_length = (max_length if max_length is not None else
                           self.tokenizer.model_max_length)
        assert (self.max_length <= self.tokenizer.model_max_length
                ), f'{self.max_length} > {self.tokenizer.model_max_length}'
        self.device = device

        self.pad_token_initialized = False
        logging.warning(f'Adding special tokens for {model_name_or_path}.')
        add_special_tokens_if_missing(self.tokenizer, self.model)
        self.pad_token_initialized = True

    def get_perplexity(self,
                       input_texts: Union[str, List[str]],
                       batch_size: int = None) -> Union[float, List[float]]:
        """Compute perplexity on input text(s).

        Args:
            input_texts (Union[str, List[str]]): Input text(s) to compute perplexity for.
            batch_size (int, optional): Batch size for perplexity computation.

        Returns:
            Union[float, List[float]]: Perplexity value(s) for the input text(s).
        """

        # Convert single input to list
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        batch_size = len(input_texts) if batch_size is None else batch_size
        batch_id = list(range(0, len(input_texts),
                              batch_size)) + [len(input_texts)]
        batch_id = list(zip(batch_id[:-1], batch_id[1:]))

        losses = []
        pbar = tqdm(batch_id, desc='Computing perplexity')
        for (start_idx, end_idx) in pbar:
            pbar.set_postfix({'batch': f'{start_idx}-{end_idx}'})
            input_text = input_texts[start_idx:end_idx]
            model_inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )

            if 'token_type_ids' in model_inputs:
                model_inputs.pop('token_type_ids')

            model_inputs = {
                k: v.to(self.device)
                for k, v in model_inputs.items()
            }
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                logits = outputs.logits
                if self.pad_token_initialized:
                    logits = logits[:, :, :-1]

                labels = model_inputs['input_ids']
                labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                valid_length = (shift_labels != IGNORE_INDEX).sum(dim=-1)

                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                loss = loss.view(len(outputs['logits']), -1)
                loss = torch.sum(loss, -1) / valid_length

            perplexity = loss.exp().cpu().tolist()
            losses.extend(perplexity)

        return losses[0] if len(losses) == 1 else losses


if __name__ == '__main__':
    # Parse command-line arguments
    parser = HfArgumentParser(ModelInferenceArguments)
    model_args, = parser.parse_args_into_dataclasses()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_args.device = device
    scorer = LLMPerplexity(
        cache_dir=model_args.cache_dir,
        model_name_or_path=model_args.model_name_or_path,
        use_auth_token=model_args.use_auth_token,
        trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        max_length=model_args.model_max_length,
        fp16=model_args.fp16,
        device=model_args.device,
    )
    text = [
        'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
        'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.',
        'I dropped my laptop on my knee, and someone stole my coffee. I am sad.',
        'I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
        'I dropped my laptop on my knee, and someone stole my coffee. I am sad.',
        'I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
        'I dropped my laptop on my knee, and someone stole my coffee. I am sad.',
        'I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
        'I dropped my laptop on my knee, and someone stole my coffee. I am sad.',
        'I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
    ]
    print(scorer.get_perplexity(text, batch_size=2))
