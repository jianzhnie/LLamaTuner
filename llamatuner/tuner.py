from typing import Any, Dict, Optional

from llamatuner.configs.parser import get_train_args
from llamatuner.train.train_full import run_full_sft
from llamatuner.train.train_lora import run_lora_sft


def run_exp(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args))
    if finetuning_args.stage == 'full_sft':
        run_full_sft(
            model_args,
            data_args,
            training_args,
            finetuning_args,
            generating_args,
        )
    elif finetuning_args.stage == 'lora_sft':
        run_lora_sft(
            model_args,
            data_args,
            training_args,
            finetuning_args,
            generating_args,
        )
    else:
        raise ValueError('Unknown task: {}.'.format(finetuning_args.stage))


def launch():
    run_exp()


if __name__ == '__main__':
    launch()
