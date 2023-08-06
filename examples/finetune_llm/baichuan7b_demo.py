import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(load_in_8bit=True, model_path=''):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True)
    inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
    inputs = inputs.to('cuda:0')
    pred = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))


if __name__ == '__main__':
    load_in_8bit = True
    model_path = '/home/robin/work_dir/llm/llm_pretrain_model/baichuan'
    main(load_in_8bit, model_path)
