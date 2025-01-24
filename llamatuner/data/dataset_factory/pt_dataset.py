from typing import Any, Dict, List

from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from llamatuner.configs import DataArguments


class PretrainDataset(Dataset):

    def __init__(self, examples: Dict[str, List[Any]],
                 tokenizer: PreTrainedTokenizer, data_args: DataArguments):
        """
        Initialize PretrainDataset with lazy loading.

        Args:
            examples: Dictionary containing the dataset examples
            tokenizer: Tokenizer for text processing
            data_args: Data arguments containing configuration
        """
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.raw_examples = examples['_prompt']
        self.eos_token = '<|end_of_text|>' if data_args.template == 'llama3' else tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.raw_examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        # Process single example on-the-fly
        messages = self.raw_examples[idx]
        text = messages[0]['content'] + self.eos_token

        if self.data_args.template == 'gemma':
            text = self.tokenizer.bos_token + text

        processed = self.tokenizer(text,
                                   add_special_tokens=False,
                                   truncation=True,
                                   max_length=self.data_args.cutoff_len)

        return {k: processed[k] for k in processed.keys()}
