import argparse
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

from chatllms.data.data_utils import IGNORE_INDEX
from chatllms.data.sft_dataset import SupervisedDataset


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
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 generation_config: argparse.Namespace, logger: None):
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.logger = logger

        # Define input prompts to generate text from
        self.sample_inputs = [
            '用一句话描述地球为什么是独一无二的。', '中国是否应该推出刺激政策救楼市？', '如何更好地融入新工作圈子',
            '帮我把这段文字转换成鲁迅作品里的语气：昨天上午，算几个数学问题时越算越难受，有想要撕掉草稿纸的冲动思维也变得缓慢，见字忘意，感觉大脑里是一团浆糊，阻力很大。\n'
            '我怀疑自己抑郁又犯了，站起身离开了书桌。走出大门，开始跑步，运动，希望能借此缓解。我不想再吃药，我担心不吃药是否能恢复。稍微运动后，大吃了一顿，路上不停的对自己说，我可以。\n',
            '回来后，感觉似乎确实好一些。', '给我写一篇大模型的新闻稿', '你觉得人类哪些工作岗位会被AI替代？',
            '请帮我写一封中式婚礼请帖，用于邀请亲朋好友参加我的婚礼！',
            '帮我写一篇八百字以上的作文，主题是：当代青年面对时代的挑战如何肩负起民族复兴的伟大任务',
            '请仿照李荣浩的风格写一首表现爱情的歌曲，以“辣椒酱”为题。', '秦王朝时期十大将军是？其主要功绩是什么？',
            '帮我写一段广告，关于房产销售的，我们的房子首付低，赠送面积大，还免两年物业费！',
            '请帮我设计一个时长为3天的北京旅游行程，行程的内容不要太紧凑，使用地铁作为交通工具，并前往前门、天安门、天坛公园、鸟巢游览，同时预留一天的时间游玩环球影城。',
            '一个笼子里面有若干只鸡和兔子，总共有50只脚和18个头，求鸡和兔子各有多少只？',
            '生成一篇短篇小说，故事情节为一个年轻人在旅途中遇到了一位神秘的老人，老人告诉他一个令人意想不到的秘密，最终年轻人的生活因此发生了翻天覆地的变化。',
            '导师想要我论文的一作，我应该怎么办？', '我现在很无聊，可以讲点有趣的事情吗？',
            '一项工程，甲、乙两队合作20天完成，乙丙两队合作60天完成，丙丁两队合作30完成，甲丁合作多少天完成?',
            '如果一位孕妇走上了公交车，但是车上没有空位了。请模拟一位热心乘客给孕妇让座的对话。',
            '桃花潭水深千尺，不及汪伦送我情。体现的是怎样的心情？',
            '编写一个简单的自动化脚本，用于批量操作文件或目录。脚本功能可以自由选择，如复制、压缩、重命名、删除等。脚本语言可使用Python、Shell、Perl等，代码长度不少于100行。',
            '音乐可以洗涤人的灵魂吗？'
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
        logger = self.logger
        logger.info('Generating sample text during evaluation...')

        # Check if the pre-trained language model is available
        if 'model' in kwargs:
            model = kwargs['model']

            # Generate text for each input prompt
            for instruction in self.sample_inputs:
                # Preprocess input prompt and convert to tensor
                inputs = f'### Instruction:\n{instruction}\n\n### Response: '
                inputs = self.tokenizer(inputs, return_tensors='pt')
                inputs = inputs.to(model.device)

                # Generate text from input prompt
                generation_output = model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                )

                # Decode generated text and log it
                generated_text = self.tokenizer.decode(generation_output[0])
                logger.info(f'Input prompt: {instruction}')
                logger.info(f'Generated text: {generated_text}')

        else:
            logger.info(
                'Pre-trained language model not found in kwargs, skipping.')


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
