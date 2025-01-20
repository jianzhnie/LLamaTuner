from enum import Enum, unique
from typing import List, Optional, TypedDict, Union

from datasets import (Dataset, IterableDataset, concatenate_datasets,
                      interleave_datasets)
from transformers import TrainingArguments

from llamatuner.configs import DataArguments
from llamatuner.utils.logger_utils import get_logger

logger = get_logger('llamatuner')


@unique
class Role(str, Enum):
    """Enumeration of possible roles in a conversation."""
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    FUNCTION = 'function'
    OBSERVATION = 'observation'


class DatasetModule(TypedDict):
    """Type definition for dataset module containing train and evaluation datasets."""
    train_dataset: Optional[Union[Dataset, IterableDataset]]
    eval_dataset: Optional[Union[Dataset, IterableDataset]]


def merge_dataset(
    all_datasets: List[Union[Dataset, IterableDataset]],
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Union[Dataset, IterableDataset]:
    """Merge multiple datasets using specified strategy.

    Args:
        all_datasets: List of datasets to merge
        data_args: Data configuration arguments
        training_args: Training configuration arguments

    Returns:
        Merged dataset

    Raises:
        ValueError: If mixing strategy is unknown
    """
    if not all_datasets:
        raise ValueError('Cannot merge empty dataset list')

    if len(all_datasets) == 1:
        return all_datasets[0]

    valid_strategies = {'concat', 'interleave_under', 'interleave_over'}
    if data_args.mix_strategy not in valid_strategies:
        raise ValueError(
            f'Unknown mixing strategy: {data_args.mix_strategy}. '
            f"Valid strategies are: {', '.join(valid_strategies)}")

    logger.info(
        f'Merging {len(all_datasets)}  datasets with {data_args.mix_strategy} strategy ...'
    )

    if data_args.mix_strategy == 'concat':
        if data_args.streaming:
            logger.warning(
                'The samples between different datasets will not be mixed in streaming mode.'
            )
        return concatenate_datasets(all_datasets)

    # Handle interleave strategies
    if not data_args.streaming:
        logger.warning(
            'We recommend using `mix_strategy=concat` in non-streaming mode.')

    stopping_strategy = 'first_exhausted' if data_args.mix_strategy == 'interleave_under' else 'all_exhausted'

    return interleave_datasets(
        datasets=all_datasets,
        probabilities=data_args.interleave_probs,
        seed=training_args.seed,
        stopping_strategy=stopping_strategy,
    )


def split_dataset(
    dataset: Union[Dataset, IterableDataset],
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> DatasetModule:
    """Split dataset into training and evaluation sets.

    Args:
        dataset: Input dataset to split
        data_args: Data configuration arguments
        training_args: Training configuration arguments

    Returns:
        Dictionary containing train and evaluation datasets

    Raises:
        ValueError: If eval_dataset_size is invalid
    """
    if data_args.eval_dataset_size <= 0:
        raise ValueError('eval_dataset_size must be greater than 0')

    val_size = int(
        data_args.eval_dataset_size
    ) if data_args.eval_dataset_size > 1 else data_args.eval_dataset_size

    logger.info(
        f'Splitting dataset with evaluation size of {val_size} '
        f'({data_args.eval_dataset_size} {"samples" if data_args.eval_dataset_size > 1 else "fraction"})'
    )
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size,
                                  seed=training_args.seed)
        val_set = dataset.take(int(data_args.eval_dataset_size))
        train_set = dataset.skip(int(data_args.eval_dataset_size))
        return DatasetModule(train_dataset=train_set, eval_dataset=val_set)

    dataset_split = dataset.train_test_split(test_size=val_size,
                                             seed=training_args.seed)
    return DatasetModule(train_dataset=dataset_split['train'],
                         eval_dataset=dataset_split['test'])
