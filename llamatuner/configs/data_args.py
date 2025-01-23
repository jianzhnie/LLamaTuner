from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    template: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Which template to use for constructing prompts in training and inference.'
        },
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The name of provided dataset(s) to use. Use commas to separate multiple datasets.'
        },
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets.'
        },
    )
    dataset_dir: str = field(
        default='data',
        metadata={'help': 'Path to the folder containing the datasets.'},
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Path to the folder containing the images or videos. Defaults to `dataset_dir`.'
        },
    )
    cutoff_len: int = field(
        default=1024,
        metadata={
            'help': 'The cutoff length of the tokenized inputs in the dataset.'
        },
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={'help': 'Whether to disable the mask on the prompt or not.'},
    )
    mask_history: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to mask the history and train on the last turn only.'
        },
    )
    streaming: bool = field(
        default=False,
        metadata={'help': 'Enable dataset streaming.'},
    )
    buffer_size: int = field(
        default=16384,
        metadata={
            'help':
            'Size of the buffer to randomly sample examples from in dataset streaming.'
        },
    )
    mix_strategy: Literal[
        'concat', 'interleave_under', 'interleave_over'] = field(
            default='concat',
            metadata={
                'help':
                'Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling).'
            },
        )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Probabilities to sample data from datasets. Use commas to separate multiple datasets.'
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            'help': 'Overwrite the cached training and evaluation sets.'
        },
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={
            'help': 'The number of examples in one group in pre-processing.'
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of processes to use for the pre-processing.'
        },
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'For debugging purposes, truncate the number of examples for each dataset.'
        },
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'Number of beams to use for evaluation. This argument will be passed to `model.generate`'
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            'help':
            'Whether or not to ignore the tokens corresponding to padded labels in the loss computation.'
        },
    )
    # 验证数据集的尺寸，也就是数量
    eval_dataset_size: Optional[float] = field(
        default=0,
        metadata={
            'help':
            'Size of the development set, should be an integer or a float in range `[0,1)`.'
        },
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Whether or not to pack the sequences in training. Will automatically enable in pre-training.'
        },
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Tool format to use for constructing function calling examples.'
        },
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={
            'help':
            ('Path to save or load the tokenized datasets. '
             'If tokenized_path not exists, it will save the tokenized datasets. '
             'If tokenized_path exists, it will load the tokenized datasets.')
        },
    )

    def __post_init__(self):

        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(',')]
            return arg

        if self.image_dir is None:
            self.image_dir = self.dataset_dir

        if self.dataset is None and self.eval_dataset_size > 0:
            raise ValueError(
                'Cannot specify `eval_dataset_size` if `dataset` is None.')

        if self.eval_dataset is not None and self.eval_dataset_size > 0:
            raise ValueError(
                'Cannot specify `eval_dataset_size` if `eval_dataset` is not None.'
            )

        if self.interleave_probs is not None:
            if self.mix_strategy == 'concat':
                raise ValueError(
                    '`interleave_probs` is only valid for interleaved mixing.')

            self.interleave_probs = list(
                map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(
                    self.interleave_probs):
                raise ValueError(
                    'The length of dataset and interleave probs should be identical.'
                )

            if self.eval_dataset is not None and len(self.eval_dataset) != len(
                    self.interleave_probs):
                raise ValueError(
                    'The length of eval dataset and interleave probs should be identical.'
                )

        if self.streaming and self.eval_dataset_size > 1e-6 and self.eval_dataset_size < 1:
            raise ValueError('Streaming mode should have an integer val size.')

        if self.streaming and self.max_samples is not None:
            raise ValueError('`max_samples` is incompatible with `streaming`.')

        if self.mask_history and self.train_on_prompt:
            raise ValueError(
                '`mask_history` is incompatible with `train_on_prompt`.')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
