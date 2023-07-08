import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase)


# 将数据保存为 JSON 文件
def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class CEval(object):
    """Class for evaluating multiple-choice questions.

    TASK2DESC: A dictionary mapping task names to their descriptions.

    args:
        model: Pre-trained model for question answering.
        tokenizer: Tokenizer for encoding text.
        data_path: Path to the dataset.
        output_dir: Directory path to save the evaluation results.


    run: Run the evaluation for all tasks.
    run_single_task: Run the evaluation for a single task.
    build_example: Builds an example string based on the given data.
    """
    TASK2DESC = {
        'high_school_physics': '高中物理',
        'fire_engineer': '注册消防工程师',
        'computer_network': '计算机网络',
        'advanced_mathematics': '高等数学',
        'logic': '逻辑学',
        'middle_school_physics': '初中物理',
        'clinical_medicine': '临床医学',
        'probability_and_statistics': '概率统计',
        'ideological_and_moral_cultivation': '思想道德修养与法律基础',
        'operating_system': '操作系统',
        'middle_school_mathematics': '初中数学',
        'chinese_language_and_literature': '中国语言文学',
        'electrical_engineer': '注册电气工程师',
        'business_administration': '工商管理',
        'high_school_geography': '高中地理',
        'modern_chinese_history': '近代史纲要',
        'legal_professional': '法律职业资格',
        'middle_school_geography': '初中地理',
        'middle_school_chemistry': '初中化学',
        'high_school_biology': '高中生物',
        'high_school_chemistry': '高中化学',
        'physician': '医师资格',
        'high_school_chinese': '高中语文',
        'tax_accountant': '税务师',
        'high_school_history': '高中历史',
        'mao_zedong_thought': '毛泽东思想和中国特色社会主义理论概论',
        'high_school_mathematics': '高中数学',
        'professional_tour_guide': '导游资格',
        'veterinary_medicine': '兽医学',
        'environmental_impact_assessment_engineer': '环境影响评价工程师',
        'basic_medicine': '基础医学',
        'education_science': '教育学',
        'urban_and_rural_planner': '注册城乡规划师',
        'middle_school_biology': '初中生物',
        'plant_protection': '植物保护',
        'middle_school_history': '初中历史',
        'high_school_politics': '高中政治',
        'metrology_engineer': '注册计量师',
        'art_studies': '艺术学',
        'college_economics': '大学经济学',
        'college_chemistry': '大学化学',
        'law': '法学',
        'sports_science': '体育学',
        'civil_servant': '公务员',
        'college_programming': '大学编程',
        'middle_school_politics': '初中政治',
        'teacher_qualification': '教师资格',
        'computer_architecture': '计算机组成',
        'college_physics': '大学物理',
        'discrete_mathematics': '离散数学',
        'marxism': '马克思主义基本原理',
        'accountant': '注册会计师',
    }

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        data_path: str = 'ceval/ceval-exam',
        output_dir: str = 'ceval_output',
    ) -> None:
        """
        Initialize the CEval object.

        Args:
            model (PreTrainedModel): Pre-trained model for question answering.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for encoding text.
            data_path (str): Path to the dataset.
            output_dir (str): Directory path to save the evaluation results.
        """
        self.model = model
        self.tokenizer = tokenizer
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.data_path = data_path
        self.output_dir = output_dir

    def run(self, shot: int, split: str) -> None:
        """
        Run the evaluation for all tasks.

        Args:
            shot (int): Number of additional examples to include in the prompt.
            split (str): Split of the dataset to evaluate on.
        """
        results: Dict[str, List[Dict[str, str]]] = {}
        accs: Dict[str, float] = {}

        # Run all tasks
        for task_name in self.TASK2DESC:
            print('=' * 100)
            print(f'Running task: {task_name}')
            result, acc = self.run_single_task(task_name, shot, split)
            results[task_name] = result
            accs[task_name] = acc
            result_path = os.path.join(self.output_dir, f'{task_name}.json')
            json_dump(result, result_path)
            print(f'Save result to {result_path}')

        # Save overall results
        acc_path = os.path.join(self.output_dir, 'acc.json')
        json_dump(accs, acc_path)
        average_acc = sum(accs.values()) / len(accs)
        print(f'Average accuracy: {average_acc}')
        results_path = os.path.join(self.output_dir, 'results.json')
        json_dump(results, results_path)
        print(f'Save results to {results_path}')

    def run_single_task(self, task_name: str, shot: int,
                        split: str) -> Tuple[List[Dict[str, str]], float]:
        """
        Run the evaluation for a single task.

        Args:
            task_name (str): Name of the task.
            shot (int): Number of additional examples to include in the prompt.
            split (str): Split of the dataset to evaluate on.

        Returns:
            Tuple containing the evaluation results and accuracy.
        """
        dataset = load_dataset(self.data_path, task_name)
        results: List[Dict[str, str]] = []
        acc = 0.0

        for data in tqdm(dataset[split]):
            prompt = f'以下是中国关于{self.TASK2DESC[task_name]}考试的单项选择题，请选出其中的正确答案。\n'
            if shot != 0:
                shuffled = dataset['dev'].shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += '\n' + self.build_example(shuffled[i],
                                                        with_answer=True)
            prompt += '\n' + self.build_example(data, with_answer=False)
            input_ids = self.tokenizer.encode(prompt,
                                              return_tensors='pt').cuda()
            output = self.model.generate(
                input_ids,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                temperature=0.1,
                top_p=0.5,
                repetition_penalty=1.1,
            )
            scores = output.scores[0][0].to(torch.float32)
            label_score = []
            candidates = ['A', 'B', 'C', 'D']
            for can in candidates:
                can_id = self.tokenizer.encode(can)[-1]
                label_score.append(scores[can_id].item())
            answer = candidates[np.argmax(label_score)]
            results.append({
                'prompt': prompt,
                'correct': answer == data['answer'].strip().upper(),
                'answer': answer,
            })
            acc += int(answer == data['answer'].strip().upper())

        acc /= len(dataset[split])
        return results, acc

    def build_example(self,
                      data: Dict[str, str],
                      with_answer: bool = True) -> str:
        """
        Builds an example string based on the given data.

        Args:
            data (Dict[str, str]): A dictionary containing the question, choices, and answer.
            with_answer (bool): Flag to include the answer in the output. Default is True.

        Returns:
            str: The formatted example string.

        Raises:
            KeyError: If the required keys are not present in the data dictionary.
        """
        # Retrieve the question from the data dictionary
        question = data['question']

        # Construct the choices by concatenating each choice with its corresponding text
        choice = '\n'.join([
            f'A. {data["A"]}',
            f'B. {data["B"]}',
            f'C. {data["C"]}',
            f'D. {data["D"]}',
        ])

        # Retrieve the answer from the data dictionary and format it
        answer = data['answer'].strip().upper() if with_answer else ''

        # Return the formatted example string
        return f'{question}\n{choice}\n答案：{answer}'


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path',
                        type=str,
                        required=True,
                        help='model name or path')
    parser.add_argument('--shot',
                        type=int,
                        default=5,
                        help='number of shot for few-shot learning')
    parser.add_argument('--split',
                        type=str,
                        default='val',
                        help='split of dataset to evaluate')
    parser.add_argument('--data_path',
                        type=str,
                        default='ceval/ceval-exam',
                        help='path to dataset')
    parser.add_argument('--output_dir',
                        type=str,
                        default='ceval_output',
                        help='output directory')
    return parser.parse_args()


def main():
    args = parse_argument()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=False,
        add_eos_token=False,
        padding_side='left',
    )
    ceval = CEval(model, tokenizer, args.data_path, args.output_dir)
    ceval.run(args.shot, args.split)


if __name__ == '__main__':
    main()
