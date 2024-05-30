import inspect
import os
from typing import Literal, Optional, Union

from datasets import Dataset, IterableDataset, load_dataset, load_from_disk
from transformers import ProcessorMixin, Seq2SeqTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer

from chatllms.configs import DataArguments, ModelArguments
from chatllms.data.data_align import align_dataset
from chatllms.data.data_parser import DatasetAttr, get_dataset_list
from chatllms.data.preprocess import get_preprocess_and_print_func
from chatllms.data.template import get_template_and_fix_tokenizer
from chatllms.data.utils import merge_dataset
from chatllms.utils.constants import FILEEXT2TYPE
from chatllms.utils.logger_utils import get_logger
from chatllms.utils.misc import has_tokenized_data

logger = get_logger(__name__)


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
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(
                        file_name.split('.')[-1], None)
                elif data_path != FILEEXT2TYPE.get(
                        file_name.split('.')[-1], None):
                    raise ValueError('File types should be identical.')
        # Check if the path is a file
        elif os.path.isfile(local_path):
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split('.')[-1], None)
        else:
            raise ValueError(f'File {local_path} not found.')
        if data_path is None:
            raise ValueError('File extension must be txt, csv, json or jsonl.')
    else:
        raise NotImplementedError('Unsupported dataset source.')

    # Load dataset from Microsoft Hub
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
                split=data_args.split,
                cache_dir=cache_dir,
                token=model_args.ms_hub_token,
                use_streaming=(data_args.streaming
                               and dataset_attr.load_from != 'file'),
            )
            if isinstance(dataset, MsDataset):
                dataset = dataset.to_hf_dataset()
        except ImportError as exc:
            raise ImportError(
                'Please install modelscope via `pip install modelscope -U`'
            ) from exc
    else:
        # Prepare arguments for `load_dataset`
        kwargs = ({
            'trust_remote_code': True
        } if 'trust_remote_code' in inspect.signature(load_dataset).parameters
                  else {})

        # Load dataset from Hugging Face Hub or local script/file
        dataset = load_dataset(
            path=data_path,
            name=dataset_attr.subset,
            data_dir=dataset_attr.folder,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=(data_args.streaming
                       and dataset_attr.load_from != 'file'),
            **kwargs,
        )

    # Convert dataset to iterable if streaming and loaded from file
    if data_args.streaming and dataset_attr.load_from == 'file':
        dataset = dataset.to_iterable_dataset()

    # Truncate dataset if max_train_samples is set
    if data_args.max_train_samples is not None:
        num_samples = min(data_args.max_train_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

    aligned_dataset = align_dataset(dataset, dataset_attr, data_args)
    return aligned_dataset


def get_dataset(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    stage: Literal['pt', 'sft', 'rm', 'kto'],
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin] = None,
) -> Union[Dataset, IterableDataset]:
    """
    Retrieves and processes the dataset for training.

    Args:
        model_args (ModelArguments): Arguments related to the model configuration.
        data_args (DataArguments): Arguments related to the dataset and data processing.
        training_args (Seq2SeqTrainingArguments): Arguments for training configuration.
        stage (Literal['pt', 'sft', 'rm', 'kto']): The current training stage.
        tokenizer (PreTrainedTokenizer): Tokenizer to be used for preprocessing.
        processor (Optional[ProcessorMixin], optional): Optional processor for additional preprocessing. Defaults to None.

    Returns:
        Union[Dataset, IterableDataset]: The processed dataset ready for training.
    """
    # Adjust the template and tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError(
            'Current template does not support `train_on_prompt`.')

    # Load tokenized dataset from disk if available
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning(
                'Loading dataset from disk will ignore other data arguments.')
            dataset = load_from_disk(data_args.tokenized_path)
            logger.info('Loaded tokenized dataset from %s.',
                        data_args.tokenized_path)
            if data_args.streaming:
                dataset = dataset.to_iterable_dataset()
            return dataset

        if data_args.streaming:
            raise ValueError(
                'Turn off `streaming` when saving dataset to disk.')

    # Load raw dataset and align it
    with training_args.main_process_first(desc='load dataset'):
        all_datasets = []
        for dataset_attr in get_dataset_list(data_args):
            if (stage == 'rm' and not dataset_attr.ranking) or (
                    stage != 'rm' and dataset_attr.ranking):
                raise ValueError(
                    'The dataset is not applicable in the current training stage.'
                )
            all_datasets.append(
                load_single_dataset(dataset_attr, model_args, data_args))

        dataset = merge_dataset(all_datasets, data_args, training_args)

    # Preprocess the dataset
    with training_args.main_process_first(desc='pre-process dataset'):
        preprocess_func, print_function = get_preprocess_and_print_func(
            data_args, training_args, stage, template, tokenizer, processor)

        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = {
                'num_proc': data_args.preprocessing_num_workers,
                'load_from_cache_file': not data_args.overwrite_cache,
                'desc': 'Running tokenizer on dataset',
            }

        dataset = dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=column_names,
            **kwargs,
        )

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
                dataset.save_to_disk(data_args.tokenized_path)
            exit(0)

        # Log a sample of the dataset
        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration as exc:
                raise RuntimeError(
                    'Cannot find valid samples, check `data/README.md` for the data format.'
                ) from exc

    return dataset
