import os
from typing import Any, Dict

from transformers import (PreTrainedModel, TrainerCallback, TrainerControl,
                          TrainingArguments)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    """Callback to save PEFT model checkpoints during training.

    Saves both the full model and the adapter model to separate directories
    within the checkpoint directory.
    """

    def save_model(self, args: Any, state: TrainingArguments,
                   kwargs: Dict[str, Any]) -> None:
        """Saves the PEFT model checkpoint.

        Args:
            args (Any): The command line arguments passed to the script.
            state (TrainingArguments): The current state of training.
            kwargs (Dict[str, Any]): A dictionary of additional keyword arguments.

        Raises:
            TypeError: If `state` is not an instance of `TrainingArguments`.
        """
        print('+' * 20, 'Saving PEFT Model Checkpoint CallBack', '+' * 20)

        # Get the checkpoint directory for saving models.
        if state.best_model_checkpoint is not None:
            # If best model checkpoint exists, use its directory as the checkpoint folder
            checkpoint_dir = os.path.join(state.best_model_checkpoint,
                                          'adapter_model')
        else:
            # Otherwise, create a new checkpoint folder using the output directory and current global step
            checkpoint_dir = os.path.join(
                args.output_dir,
                f'{PREFIX_CHECKPOINT_DIR}-{state.global_step}')

        # Create path for the PEFT model
        peft_model_path = os.path.join(checkpoint_dir, 'adapter_model')
        model: PreTrainedModel = kwargs['model']
        model.save_pretrained(peft_model_path)

        # Create path for the PyTorch model binary file and remove it if it already exists
        pytorch_model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args: Any, state: TrainingArguments,
                control: TrainerControl,
                **kwargs: Dict[str, Any]) -> TrainerControl:
        """Callback method that calls save_model() and returns `control`
        argument.

        Args:
            args (Any): The command line arguments passed to the script.
            state (TrainingArguments): The current state of training.
            control (trainer_callback.TrainerControl): \
                The current state of the TrainerCallback's control flow.
            kwargs (Dict[str, Any]): A dictionary of additional keyword arguments.

        Returns:
            trainer_callback.TrainerControl: The current state of the TrainerCallback's control flow.

        Raises:
            TypeError: If `state` is not an instance of `TrainingArguments`.
        """
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args: Any, state: TrainingArguments,
                     control: TrainerControl, **kwargs: Dict[str,
                                                             Any]) -> None:
        """Callback method that saves the model checkpoint and creates a
        'completed' file in the output directory.

        Args:
            args (Any): The command line arguments passed to the script.
            state (TrainingArguments): The current state of training.
            control (trainer_callback.TrainerControl): \
                The current state of the TrainerCallback's control flow.
            kwargs (Dict[str, Any]): A dictionary of additional keyword arguments.

        Raises:
            TypeError: If `state` is not an instance of `TrainingArguments`.
        """

        # Define a helper function to create a 'completed' file in the output directory
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        # Create the 'completed' file in the output directory
        touch(os.path.join(args.output_dir, 'completed'))
