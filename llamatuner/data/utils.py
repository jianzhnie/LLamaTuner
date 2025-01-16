from enum import Enum, unique
from typing import Dict, List, Union

from datasets import (Dataset, IterableDataset, concatenate_datasets,
                      interleave_datasets)
from transformers import TrainingArguments

from llamatuner.configs import DataArguments
from llamatuner.utils.logger_utils import get_logger

logger = get_logger('llamatuner')


@unique
class Role(str, Enum):
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    FUNCTION = 'function'
    OBSERVATION = 'observation'


def merge_dataset(
    all_datasets: List[Union[Dataset, IterableDataset]],
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Union[Dataset, IterableDataset]:
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == 'concat':
        if data_args.streaming:
            logger.warning(
                'The samples between different datasets will not be mixed in streaming mode.'
            )
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith('interleave'):
        if not data_args.streaming:
            logger.warning(
                'We recommend using `mix_strategy=concat` in non-streaming mode.'
            )
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=training_args.seed,
            stopping_strategy='first_exhausted'
            if data_args.mix_strategy.endswith('under') else 'all_exhausted',
        )
    else:
        raise ValueError('Unknown mixing strategy.')


def split_dataset(
    dataset: Union[Dataset, IterableDataset],
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict[str, Dataset]:
    if training_args.do_train:
        if data_args.eval_dataset_size > 1e-6:  # Split the dataset
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size,
                                          seed=training_args.seed)
                val_set = dataset.take(int(data_args.eval_dataset_size))
                train_set = dataset.skip(int(data_args.eval_dataset_size))
                return {'train_dataset': train_set, 'eval_dataset': val_set}
            else:
                val_size = (int(data_args.eval_dataset_size)
                            if data_args.eval_dataset_size > 1 else
                            data_args.eval_dataset_size)
                dataset = dataset.train_test_split(test_size=val_size,
                                                   seed=training_args.seed)
                return {
                    'train_dataset': dataset['train'],
                    'eval_dataset': dataset['test'],
                }
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size,
                                          seed=training_args.seed)
            return {'train_dataset': dataset}
    else:  # do_eval or do_predict
        return {'eval_dataset': dataset}
