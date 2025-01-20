import os
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np
from datasets import (Dataset, DatasetDict, IterableDataset, load_dataset,
                      load_from_disk)
from transformers import ProcessorMixin, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer

from llamatuner.configs import DataArguments, ModelArguments
from llamatuner.data.data_align import align_dataset
from llamatuner.data.data_parser import DatasetAttr, get_dataset_attr_list
from llamatuner.data.preprocess import get_preprocess_and_print_func
from llamatuner.data.template import Template, get_template_and_fix_tokenizer
from llamatuner.data.utils import DatasetModule, merge_dataset
from llamatuner.utils.constants import FILEEXT2TYPE
from llamatuner.utils.logger_utils import get_logger
from llamatuner.utils.misc import has_tokenized_data

logger = get_logger('llamatuner')


def load_single_dataset(
    dataset_attr: DatasetAttr,
    model_args: ModelArguments,
    data_args: DataArguments,
) -> Union[Dataset, IterableDataset]:
    """
    Load a single dataset based on the provided dataset attributes, model arguments, and data arguments.

    Args:
        dataset_attr (DatasetAttr): Attributes of the dataset to be loaded.
        model_args (ModelArguments): Arguments related to the model and cache directories.
        data_args (DataArguments): Arguments related to data loading and processing.
        logger (logging.Logger): Logger for logging information and errors.

    Returns:
        Union[Dataset, IterableDataset]: The loaded dataset.
    """
    logger.info('Loading dataset %s...', dataset_attr)
    data_path, data_files = None, None

    # Determine dataset source and configure paths
    if dataset_attr.load_from in ['hf_hub', 'ms_hub']:
        data_path = dataset_attr.dataset_name
    elif dataset_attr.load_from == 'script':
        data_path = os.path.join(data_args.dataset_dir,
                                 dataset_attr.dataset_name)
    elif dataset_attr.load_from == 'file':
        data_files = []
        local_path = os.path.join(data_args.dataset_dir,
                                  dataset_attr.dataset_name)
        # Check if the path is a directory
        if os.path.isdir(local_path):
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        # Check if the path is a file
        elif os.path.isfile(local_path):
            data_files.append(local_path)
        else:
            raise ValueError(f'File {local_path} not found.')
        data_path = FILEEXT2TYPE.get(
            os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError('Allowed file types: {}.'.format(','.join(
                FILEEXT2TYPE.keys())))
        if any(data_path != FILEEXT2TYPE.get(
                os.path.splitext(data_file)[-1][1:], None)
               for data_file in data_files):
            raise ValueError('File types should be identical.')
    else:
        raise NotImplementedError('Unsupported dataset source.')

    # Load dataset from ModelScope Hub
    if dataset_attr.load_from == 'ms_hub':
        try:
            from modelscope import MsDataset
            from modelscope.utils.config_ds import MS_DATASETS_CACHE

            cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
            dataset = MsDataset.load(
                dataset_name=data_path,
                subset_name=dataset_attr.subset,
                data_dir=dataset_attr.folder,
                data_files=data_files,
                split=dataset_attr.split,
                cache_dir=cache_dir,
                token=model_args.ms_hub_token,
                use_streaming=data_args.streaming,
            )
            if isinstance(dataset, MsDataset):
                dataset = dataset.to_hf_dataset()
        except ImportError as exc:
            raise ImportError(
                'Please install modelscope via `pip install modelscope -U`'
            ) from exc
    else:
        # Load dataset from Hugging Face Hub or local script/file
        dataset = load_dataset(
            path=data_path,
            name=dataset_attr.subset,
            data_dir=dataset_attr.folder,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
        )

    logger.info(f'Successfully loaded dataset {dataset_attr.dataset_name}')
    if dataset_attr.num_samples is not None and not data_args.streaming:
        dataset_size = len(dataset)
        target_num = dataset_attr.num_samples

        # If target samples exceed dataset size, use sampling with replacement
        if target_num > dataset_size:
            indexes = np.concatenate([
                np.random.permutation(dataset_size),
                np.random.choice(dataset_size, target_num - dataset_size)
            ])
        else:
            indexes = np.random.permutation(dataset_size)[:target_num]

        dataset = dataset.select(indexes)
        logger.info(
            f'Sampled {target_num} examples from dataset {dataset_attr}.')
    # Truncate dataset if max_train_samples is set
    if data_args.max_samples is not None:
        num_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(num_samples))
        logger.info(
            f'Sampled {data_args.max_samples} examples from dataset {dataset_attr}.'
        )

    logger.info(
        f'Aligning the dataset to the {dataset_attr.formatting} template.')
    aligned_dataset = align_dataset(dataset, dataset_attr, data_args)
    logger.info(
        f'Successfully converted dataset {dataset_attr.dataset_name} to {dataset_attr.formatting} format.'
    )
    return aligned_dataset


def get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    stage: Literal['pt', 'sft', 'rm', 'ppo', 'kto'],
) -> Optional[Union[Dataset, IterableDataset]]:
    """
    Merge multiple datasets into a single dataset.

    Args:
        dataset_names: List of dataset names to merge.
        data_args: Arguments related to data loading and processing.
        model_args: Arguments related to model configuration.
        training_args: Arguments for training configuration.
        stage: Current training stage.

    Returns:
        Optional[Union[Dataset, IterableDataset]]: Merged dataset or None if no datasets provided.
    """
    if dataset_names is None:
        return None

    all_datasets = []
    for dataset_attr in get_dataset_attr_list(dataset_names, data_args):
        if (stage == 'rm'
                and not dataset_attr.ranking) or (stage != 'rm'
                                                  and dataset_attr.ranking):
            raise ValueError(
                'The dataset is not applicable in the current training stage.')
        single_dataset = load_single_dataset(dataset_attr, model_args,
                                             data_args)
        all_datasets.append(single_dataset)

    dataset = merge_dataset(all_datasets, data_args, training_args)
    return dataset


def get_preprocessed_dataset(
    dataset: Optional[Union[Dataset, IterableDataset]],
    data_args: DataArguments,
    training_args: TrainingArguments,
    stage: Literal['pt', 'sft', 'rm', 'ppo', 'kto'],
    template: Template,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin] = None,
    is_eval: bool = False,
) -> Optional[Union[Dataset, IterableDataset]]:
    """
    Preprocess the dataset by applying tokenization and formatting.

    Args:
        dataset: Input dataset to preprocess.
        data_args: Arguments related to data processing.
        training_args: Arguments for training configuration.
        stage: Current training stage.
        template: Template for data formatting.
        tokenizer: Tokenizer for text processing.
        processor: Optional additional processor.
        is_eval: Whether this is evaluation dataset.

    Returns:
        Optional[Union[Dataset, IterableDataset]]: Preprocessed dataset or None if input is None.

    Raises:
        RuntimeError: If insufficient or invalid samples are found.
    """
    if dataset is None:
        return None

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, stage, template, tokenizer, processor, do_generate=False)

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}

    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache)
            or (training_args.local_process_index != 0),
            desc='Running tokenizer on dataset',
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            logger.info('eval example:' if is_eval else 'training example:')
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == 'pt':
                raise RuntimeError(
                    'Cannot find sufficient samples, consider increasing dataset size.'
                )
            else:
                raise RuntimeError(
                    'Cannot find valid samples, check `data/README.md` for the data format.'
                )

    return dataset


def get_dataset(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    stage: Literal['pt', 'sft', 'rm', 'ppo', 'kto'],
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin] = None,
) -> DatasetModule:
    """
    Retrieves and processes the dataset for training.

    Args:
        data_args (DataArguments): Arguments related to the dataset and data processing.
        model_args (ModelArguments): Arguments related to the model configuration.
        training_args (TrainingArguments): Arguments for training configuration.
        stage (Literal['pt', 'sft', 'rm', 'ppo', 'kto']): The current training stage.
        tokenizer (PreTrainedTokenizer): Tokenizer to be used for preprocessing.
        processor (Optional[ProcessorMixin], optional): Optional processor for additional preprocessing. Defaults to None.

    Returns:
        Union[Dataset, IterableDataset]: The processed dataset ready for training.
    """
    # Adjust the template and tokenizer
    logger.info('Get template and fix tokenizer')
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    logger.info('Using template: %s', template)

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError(
            'Current template does not support `train_on_prompt`.')

    # Load tokenized dataset from disk if available
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning(
                'Loading dataset from disk will ignore other data arguments.')
            tokenized_data = load_from_disk(data_args.tokenized_path)
            logger.info('Loaded tokenized dataset from %s.',
                        data_args.tokenized_path)

            dataset_module: Dict[str, Dataset] = {}
            if isinstance(tokenized_data, DatasetDict):
                if 'train_dataset' in tokenized_data:
                    dataset_module['train_dataset'] = tokenized_data[
                        'train_dataset']

                if 'eval_dataset' in tokenized_data:
                    dataset_module['eval_dataset'] = tokenized_data[
                        'eval_dataset']

            else:  # Dataset
                dataset_module['train_dataset'] = tokenized_data

            if data_args.streaming:
                dataset_module = {
                    k: v.to_iterable_dataset()
                    for k, v in dataset_module.items()
                }
            return dataset_module

        if data_args.streaming:
            raise ValueError(
                'Turn off `streaming` when saving dataset to disk.')

    # Load raw dataset and align it
    with training_args.main_process_first(desc='load dataset'):
        train_dataset_names = ([
            ds.strip() for ds in data_args.dataset.split(',')
        ] if data_args.dataset else [])
        eval_dataset_names = ([
            ds.strip() for ds in data_args.eval_dataset.split(',')
        ] if data_args.eval_dataset else [])
        logger.info(f'Train dataset names: {train_dataset_names}')
        logger.info(f'Eval dataset names: {eval_dataset_names}')
        train_dataset = get_merged_dataset(train_dataset_names, data_args,
                                           model_args, training_args, stage)
        eval_dataset = get_merged_dataset(eval_dataset_names, data_args,
                                          model_args, training_args, stage)

    # Preprocess the dataset
    with training_args.main_process_first(desc='pre-process dataset'):

        train_dataset = get_preprocessed_dataset(train_dataset,
                                                 data_args,
                                                 training_args,
                                                 stage,
                                                 template,
                                                 tokenizer,
                                                 processor,
                                                 is_eval=False)
        eval_dataset = get_preprocessed_dataset(eval_dataset,
                                                data_args,
                                                training_args,
                                                stage,
                                                template,
                                                tokenizer,
                                                processor,
                                                is_eval=True)

        dataset_dict = {}
        if train_dataset is not None:
            if data_args.streaming:
                train_dataset = train_dataset.shuffle(
                    buffer_size=data_args.buffer_size, seed=training_args.seed)

            dataset_dict['train_dataset'] = train_dataset

        if eval_dataset is not None:
            if data_args.streaming:
                eval_dataset = eval_dataset.shuffle(
                    buffer_size=data_args.buffer_size, seed=training_args.seed)

            dataset_dict['eval_dataset'] = eval_dataset

        dataset_dict = DatasetDict(dataset_dict)

        # Save tokenized dataset to disk if required
        if data_args.tokenized_path is not None:
            if training_args.should_save:
                logger.info(
                    'Tokenized dataset saved at %s.',
                    data_args.tokenized_path,
                )
                logger.info(
                    'Please restart the training with `--tokenized_path %s`.',
                    data_args.tokenized_path,
                )
                dataset_dict.save_to_disk(data_args.tokenized_path)
            exit(0)

        dataset_module = {}
        if 'train_dataset' in dataset_dict:
            dataset_module['train_dataset'] = dataset_dict['train_dataset']

        if 'eval_dataset' in dataset_dict:
            dataset_module['eval_dataset'] = dataset_dict['eval_dataset']

    return dataset_module
