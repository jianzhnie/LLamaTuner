from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('baichuan-inc/baichuan-7B',
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('baichuan-inc/baichuan-7B',
                                             device_map='auto',
                                             trust_remote_code=True)
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
