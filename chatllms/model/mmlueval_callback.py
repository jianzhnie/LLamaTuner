import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (PreTrainedModel, PreTrainedTokenizer, Trainer,
                          TrainerCallback)

from chatllms.data.data_utils import IGNORE_INDEX
from chatllms.data.sft_dataset import SupervisedDataset


@dataclass
class MMLUEvalCallback(TrainerCallback):
    """
    A callback function called after each evaluation step during training to evaluate \
        the performance of a model on an
    MMLU (Mean Length of Utterance) dataset.

    Args:
        trainer (Trainer): The trainer instance to be used.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model being trained.
        data_dir (str): The directory where the MMLU dataset is stored.
        args (argparse.Namespace): The command line arguments for the current run.
    """
    def __init__(
        self,
        trainer: 'Trainer',
        tokenizer: PreTrainedTokenizer,
        data_dir: str,
        args: argparse.Namespace,
    ) -> None:
        self.trainer = trainer
        self.tokenizer = tokenizer

        # Load the appropriate MMLU dataset based on the value of 'mmlu_dataset'.
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset(
                'json',
                data_files={
                    'eval':
                    os.path.join(data_dir, 'mmlu/zero_shot_mmlu_val.json'),
                    'test':
                    os.path.join(data_dir, 'mmlu/zero_shot_mmlu_test.json'),
                },
            )
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        #  MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset in ['mmlu', 'mmlu-fs']:
            mmlu_dataset = load_dataset(
                'json',
                data_files={
                    'eval':
                    os.path.join(data_dir, 'mmlu/five_shot_mmlu_val.json'),
                    'test':
                    os.path.join(data_dir, 'mmlu/five_shot_mmlu_test.json'),
                },
            )
        else:
            raise ValueError(
                f"Invalid value '{args.mmlu_dataset}' for argument 'mmlu_dataset'."
            )

        # Select the appropriate split of the dataset and limit the number of samples to evaluate.
        self.mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            self.mmlu_dataset = self.mmlu_dataset.select(
                range(args.max_mmlu_samples))

        # Define a list of token IDs representing the letters A, B, C, and D.
        self.abcd_idx = [
            tokenizer('A', add_special_tokens=False).input_ids[0],
            tokenizer('B', add_special_tokens=False).input_ids[0],
            tokenizer('C', add_special_tokens=False).input_ids[0],
            tokenizer('D', add_special_tokens=False).input_ids[0],
        ]
        # Load the accuracy metric for evaluating MMLU performance.
        self.accuracy = evaluate.load('accuracy')
        self.mmlu_dataset = SupervisedDataset(
            self.mmlu_dataset,
            tokenizer=tokenizer,
            source_max_len=args.mmlu_source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=args.train_on_source,
            predict_with_generate=args.predict_with_generate,
        )

    def on_evaluate(
        self,
        args: Dict[str, Any],
        state: Dict[str, Any],
        control: Dict[str, Any],
        model: PreTrainedModel,
        **kwargs: Any,
    ) -> None:
        """
        Iterate over the batches of the evaluation dataset and make predictions for MMLU.

        Args:
            args (Dict[str, Any]): Dictionary containing the evaluation arguments.
            state (Dict[str, Any]): Dictionary containing the current state of the trainer.
            control (Dict[str, Any]): Dictionary containing the evaluation control variables.
            model (PreTrainedModel): The model being evaluated.
        """
        # Get the evaluation data loader and set the maximum length of the source sequence.

        data_loader = self.trainer.get_eval_dataloader(self.mmlu_dataset)

        # Set the trainer model in evaluation mode and initialize empty lists for predictions and references.
        self.trainer.model.eval()
        preds, refs = [], []
        loss_mmlu = 0

        # Iterate over the batches of the evaluation dataset and make predictions.
        for batch in tqdm(data_loader, total=len(data_loader)):
            (loss, logits, labels) = self.trainer.prediction_step(
                self.trainer.model,
                batch,
                prediction_loss_only=False,
            )

            # Extract the predictions for A, B, C, and D tokens.
            for i, logit in enumerate(logits):
                label_non_zero_id = (batch['labels'][i] !=
                                     IGNORE_INDEX).nonzero()[0][0]

                logit_abcd = logit[label_non_zero_id - 1][self.abcd_idx]
                preds.append(torch.argmax(logit_abcd).item())

            # Extract the ground truth labels and compute the accuracy by subject.
            labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
            refs += [self.abcd_idx.index(label) for label in labels.tolist()]
            loss_mmlu += loss.item()

        # Extract results by subject.
        results = {'mmlu_loss': loss_mmlu / len(data_loader)}
        subject = self.mmlu_dataset['subject']
        subjects = {s: {'refs': [], 'preds': []} for s in set(subject)}
        for s, p, r in zip(subject, preds, refs):
            subjects[s]['preds'].append(p)
            subjects[s]['refs'].append(r)

        # Compute the accuracy score for each subject and log the results.
        subject_scores = []
        for subject in subjects:
            subject_score = self.accuracy.compute(
                references=subjects[subject]['refs'],
                predictions=subjects[subject]['preds'])['accuracy']
            results[
                f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
            subject_scores.append(subject_score)

        # Compute the overall MMLU accuracy and log the results.
        results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
        self.trainer.log(results)
