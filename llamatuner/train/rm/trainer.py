import json
import os
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, Trainer
from transformers.trainer import PredictionOutput

from llamatuner.utils.logger_utils import get_logger

logger = get_logger('llamatuner')


class PairwiseTrainer(Trainer):
    r"""
    Inherits Trainer to compute pairwise loss. This custom Trainer computes a pairwise ranking loss
    where the first half of the batch contains positive examples and the second half contains negative examples.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.can_return_loss = True

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        Args:
            model (PreTrainedModel): The model being trained.
            inputs (Dict[str, torch.Tensor]): The inputs and targets of the model.
            return_outputs (bool, optional): Whether to return the model outputs in addition to the loss. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: The computed loss and optionally the model outputs.
        """
        outputs = model(
            **inputs, output_hidden_states=True, return_dict=True, use_cache=False
        )
        values = outputs.logits

        batch_size = inputs["input_ids"].size(0) // 2
        chosen_masks, rejected_masks = torch.split(
            inputs["attention_mask"], batch_size, dim=0
        )
        chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)

        chosen_scores = chosen_rewards.gather(
            dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1)
        )
        rejected_scores = rejected_rewards.gather(
            dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1)
        )

        chosen_scores, rejected_scores = (
            chosen_scores.squeeze(),
            rejected_scores.squeeze(),
        )

        loss = -F.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()
        if return_outputs:
            return loss, (chosen_scores, rejected_scores)
        else:
            return loss

    def save_predictions(self, predict_results: PredictionOutput) -> None:
        r"""
        Saves model predictions to `output_dir`.

        Args:
            predict_results (PredictionOutput): The output of the `predict` method.

        This method saves the chosen and rejected scores to a JSONL file in the `output_dir`.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(
            self.args.output_dir, "generated_predictions.jsonl"
        )
        logger.info(f"Saving prediction results to {output_prediction_file}")
        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(
                    json.dumps(
                        {
                            "chosen": round(float(c_score), 2),
                            "rejected": round(float(r_score), 2),
                        }
                    )
                )

            writer.write("\n".join(res))
