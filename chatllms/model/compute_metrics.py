from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import jieba
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from transformers import PreTrainedTokenizer


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
