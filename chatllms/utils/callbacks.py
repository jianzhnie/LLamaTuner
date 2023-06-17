import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import evaluate
import jieba
import numpy as np
import torch
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from tqdm.auto import tqdm
from transformers import (PreTrainedModel, PreTrainedTokenizer, Trainer,
                          TrainerCallback)

from .data_utils import IGNORE_INDEX


@dataclass
class ComputeMetrics:
    """
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    Borrowed from: https://github.com/THUDM/ChatGLM-6B/blob/0c2806fea82683349194e21996dd6b3acc3c265b/ptuning/main.py#L307
    
    """
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """
        Initialize the ComputeMetrics class with a pre-trained tokenizer object.

        Args:
            tokenizer (PreTrainedTokenizer): A pre-trained tokenizer object to be used for decoding tokenized sequences.
        """
        self.tokenizer = tokenizer

    def __call__(
        self, eval_preds: List[Union[np.ndarray, Tuple[np.ndarray]]]
    ) -> Dict[str, float]:
        """
        Computes evaluation metrics for model predictions.

        Args:
            eval_preds (List[Union[np.ndarray, Tuple[np.ndarray]]]): List of tuples containing prediction and label arrays.

        Returns:
            Dict[str, float]: A dictionary containing the average of each computed metric over all prediction-label pairs.
        """

        # Extract predictions and labels from input
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace IGNORE_INDEX in the labels with pad_token_id as we cannot decode them if ignore_pad_token_for_loss=True.
        preds = np.where(preds != self.tokenizer.pad_token_id, preds,
                         self.tokenizer.pad_token_id)
        labels = np.where(labels != self.tokenizer.pad_token_id, labels,
                          self.tokenizer.pad_token_id)

        score_dict = {
            'rouge-1': [],  # numericl 1
            'rouge-2': [],
            'rouge-l': [],  # string l
            'bleu-4': []
        }

        # Calculate metrics for each prediction-label pair
        for pred, label in zip(preds, labels):
            pred = pred[(pred == self.tokenizer.bos_token_id
                         ).nonzero()[0][0]:]  # remove the query
            hypothesis = list(
                jieba.cut(self.tokenizer.decode(pred,
                                                skip_special_tokens=True)))
            reference = list(
                jieba.cut(
                    self.tokenizer.decode(label, skip_special_tokens=True)))

            # If there are no words in the hypothesis, set all scores to 0
            if len(' '.join(hypothesis).split()) == 0:
                result = {
                    'rouge-1': {
                        'f': 0.0
                    },
                    'rouge-2': {
                        'f': 0.0
                    },
                    'rouge-l': {
                        'f': 0.0
                    }
                }
            else:
                rouge = Rouge()
                scores = rouge.get_scores(' '.join(hypothesis),
                                          ' '.join(reference))
                result = scores[0]

            # Append scores to score_dict
            for k, v in result.items():
                score_dict[k].append(round(v['f'] * 100, 4))

            # Calculate BLEU-4 score and append it to score_dict
            bleu_score = sentence_bleu(
                [list(label)],
                list(pred),
                smoothing_function=SmoothingFunction().method3)
            score_dict['bleu-4'].append(round(bleu_score * 100, 4))

        # Calculate average of each metric over all prediction-label pairs and return as a dictionary
        return {k: float(np.mean(v)) for k, v in score_dict.items()}


@dataclass
class SampleGenerateCallback(TrainerCallback):
    """
    A callback that generates text samples from a pre-trained language model during training.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used to preprocess inputs.
        max_new_tokens (int): The maximum number of tokens to generate in response to each input.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 max_new_tokens: int = 70):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        # Define input prompts to generate text from
        self.sample_inputs = [
            '用一句话描述地球为什么是独一无二的。',
            '中国是否应该推出刺激政策救楼市？',
            '如何更好地融入新工作圈子',
        ]

    def on_evaluate(self, args: Any, state: Dict[str, Any], control: Any,
                    **kwargs: Any) -> None:
        """
        Generates text samples from the language model during evaluation.

        Args:
            args (Any): Trainer arguments, not used in this method.
            state (Dict[str, Any]): Trainer state dictionary, not used in this method.
            control (Any): Trainer control object, not used in this method.
            kwargs (Dict[str, Any]): Keyword arguments passed to the method, including the pre-trained
                language model (under the key 'model') and any additional parameters needed for generation.

        Returns:
            None
        """
        logger = logging.getLogger(__name__)
        logger.info('Generating sample text during evaluation...')

        # Check if the pre-trained language model is available
        if 'model' in kwargs:
            model = kwargs['model']

            # Generate text for each input prompt
            for sample_input in self.sample_inputs:
                # Preprocess input prompt and convert to tensor
                inputs = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{sample_input}\n\n### Response: '
                input_ids = self.tokenizer.encode(inputs, return_tensors='pt')
                input_ids = input_ids.to(model.device)

                # Generate text from input prompt
                generation_output = model.generate(
                    input_ids=input_ids,
                    max_length=len(input_ids[0]) + self.max_new_tokens,
                )

                # Decode generated text and log it
                generated_text = self.tokenizer.decode(generation_output[0])
                logger.info(f'Input prompt: {sample_input}')
                logger.info(f'Generated text: {generated_text}')

        else:
            logger.info(
                'Pre-trained language model not found in kwargs, skipping.')


@dataclass
class MMLUEvalCallback(TrainerCallback):
    """
    A callback function called after each evaluation step during training to evaluate the performance of a model on an
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
        ) if args.do_predict else None

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
