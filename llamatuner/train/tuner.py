from typing import Any, Dict, Optional

from llamatuner.configs.parser import get_train_args
from llamatuner.train.pt.train_pt import run_pt
from llamatuner.train.sft.train_full import run_full_sft
from llamatuner.train.sft.train_lora import run_lora_sft


def run_exp(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args))
    if finetuning_args.stage == 'pt':
        run_pt(
            model_args,
            data_args,
            training_args,
            finetuning_args,
        )
    if finetuning_args.stage == 'full':
        run_full_sft(
            model_args,
            data_args,
            training_args,
            finetuning_args,
            generating_args,
        )
    elif finetuning_args.stage == 'sft':
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
