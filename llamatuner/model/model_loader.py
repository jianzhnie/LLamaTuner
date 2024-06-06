from typing import Any, Dict

from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          AutoTokenizer, PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer)
from trl import AutoModelForCausalLMWithValueHead

from llamatuner.configs import FinetuningArguments, ModelArguments
from llamatuner.model.adapter import init_adapter
from llamatuner.model.patcher import (patch_config, patch_model,
                                      patch_tokenizer, patch_valuehead_model)
from llamatuner.model.utils import load_valuehead_params, register_autoclass
from llamatuner.utils.logger_utils import get_logger
from llamatuner.utils.misc import count_parameters, try_download_model_from_ms

logger = get_logger(__name__)


def get_init_kwargs(model_args: ModelArguments) -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        'trust_remote_code': True,
        'cache_dir': model_args.cache_dir,
        'revision': model_args.model_revision,
        'token': model_args.hf_hub_token,
    }


def load_config(model_args: ModelArguments) -> PretrainedConfig:
    r"""
    Loads model config.
    """
    init_kwargs = get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path,
                                      **init_kwargs)


def load_tokenizer(model_args: ModelArguments):
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side='right',
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side='right',
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )

        logger.info('Add {} to special tokens.'.format(','.join(
            model_args.new_special_tokens)))

        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning(
                'New tokens have been added, changed `resize_vocab` to True.')

    patch_tokenizer(tokenizer)

    if model_args.visual_inputs:
        try:
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path, **init_kwargs)
            setattr(processor, 'tokenizer', tokenizer)
        except Exception as exc:
            raise ValueError(
                'This multimodal LLM is not supported.\n'
                'Download LLaVA-1.5 models from: https://huggingface.co/llava-hf\n'
                'Download Yi-VL models from: https://huggingface.co/BUAADreamer'
            ) from exc
    else:
        processor = None

    return {'tokenizer': tokenizer, 'processor': processor}


def load_model(
    tokenizer: PreTrainedTokenizer,
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> PreTrainedModel:
    r"""
    Loads pretrained model.
    """
    init_kwargs = get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    lazy_load = False

    if model is None and not lazy_load:
        init_kwargs['config'] = config
        init_kwargs[
            'pretrained_model_name_or_path'] = model_args.model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(**init_kwargs)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args,
                         is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info(
                'Loaded valuehead from checkpoint: {}'.format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            'trainable params: {:d} || all params: {:d} || trainable%: {:.4f}'.
            format(trainable_params, all_param,
                   100 * trainable_params / all_param))
    else:
        param_stats = 'all params: {:d}'.format(all_param)

    logger.info(param_stats)
    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print('name: {}, dtype: {}, device: {}, trainable: {}'.format(
                name, param.dtype, param.device, param.requires_grad))
    return model
