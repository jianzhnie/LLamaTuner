import argparse
import json
import logging
import os
from typing import Any, Dict

import numpy as np
import transformers
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def train_and_evaluate(trainer: transformers.Trainer,
                       args: argparse.Namespace) -> None:
    """
    Trains and evaluates a machine learning model.

    Args:
        trainer (Trainer): The training object to use for training and evaluation.
        args (argparse.Namespace): The command line arguments for the current run.
    Returns:
        None
    """
    # Create dictionary to store metrics
    all_metrics: Dict[str, Any] = {'run_name': args.run_name}

    # Training
    if args.do_train:
        logger.info('=' * 80)
        logger.info('*** Train ***')
        logger.info('=' * 80)
        train_result = trainer.train()
        metrics = train_result.metrics

        # Log and save training metrics
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

        # Update metrics dictionary with training metrics
        all_metrics.update(metrics)

    # Evaluation
    if args.do_eval:
        logger.info('=' * 80)
        logger.info('*** Evaluate ***')
        logger.info('=' * 80)

        # Evaluate the trained model and obtain evaluation metrics
        metrics = trainer.evaluate(metric_key_prefix='eval')

        # Log and save evaluation metrics
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

        # Update metrics dictionary with evaluation metrics
        all_metrics.update(metrics)

    # Save all metrics to a json file
    if args.do_train or args.do_eval:
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fout:
            fout.write(json.dumps(all_metrics))


def predict_and_save(trainer: transformers.Trainer,
                     tokenizer: transformers.PreTrainedTokenizer,
                     predict_dataset: Dataset,
                     args: argparse.Namespace) -> None:
    """
    Make predictions on new data, save them to a file along with input examples,
    and update the overall metrics.
    """
    logger.info('=' * 80)
    logger.info('*** Predict ***')
    logger.info('=' * 80)
    data_dict = predict_dataset.dataset

    # Make predictions on the test dataset
    prediction_output = trainer.predict(test_dataset=predict_dataset,
                                        metric_key_prefix='predict')

    # Get the predictions and metrics
    prediction_metrics = prediction_output.metrics
    predictions = prediction_output.predictions

    # Replace -100 values with pad token ID and decode predictions
    predictions = np.where(predictions != -100, predictions,
                           tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(predictions,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)

    data_dict = predict_dataset.dataset
    # Create dictionary to store metrics
    all_metrics: Dict[str, Any] = {'run_name': args.run_name}
    # Write predictions and input examples to file
    with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
        for i, example in enumerate(data_dict):
            example['prediction_with_input'] = predictions[i].strip()
            example['prediction'] = predictions[i].replace(
                example['input'], '').strip()
            fout.write(json.dumps(example) + '\n')

    # Print and log the prediction metrics
    print(prediction_metrics)
    trainer.log_metrics('predict', prediction_metrics)
    trainer.save_metrics('predict', prediction_metrics)

    # Update the overall metrics
    all_metrics.update(prediction_metrics)

    # Save the overall metrics to a file
    with open(os.path.join(args.output_dir, 'eval_metrics.json'), 'w') as fout:
        fout.write(json.dumps(all_metrics))
