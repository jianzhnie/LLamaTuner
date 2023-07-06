from datasets import load_dataset
import argparse
from clean_sharegpt import json_dump, get_clean_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str)
    parser.add_argument('--out-file', type=str)
    args = parser.parse_args()
    args.in_file = '/home/robin/prompt_data/WizardLM/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.json'
    args.out_file = 'evol_instruct_clean.json'

    clean_data2 = get_clean_data(args)
    json_dump(clean_data2, args.out_file)