python clean_sharegpt.py \
    --in-file /userhome/jianzhnie/prompt_data/anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
    --out-file /userhome/jianzhnie/prompt_data/sharegpt/sharegpt_clean.json 


python split_long_conversation.py \
    --in-file  /userhome/jianzhnie/prompt_data/sharegpt/sharegpt_clean.json \
    --out-file /userhome/jianzhnie/prompt_data/sharegpt/sharegpt_split.json \
    --model-name-or-path /userhome/jianzhnie/checkpoints/llama7b 

python clean_evol_instruct.py \
    --in-file /userhome/jianzhnie/prompt_data/WizardLM/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.json \
    --out-file /userhome/jianzhnie/prompt_data/sharegpt/evol_instruct_clean.json


python chatllms/data/utils/convert_alpaca.py \
    --in-file ./prompt_data/chinese_data/alpaca_data_zh_51k.json \
    --out-file ./prompt_data/chinese_data/alpaca_vicuna.json


python chatllms/data/utils/convert_alpaca.py \
    --in-file ./prompt_data/InstructionWild/instinwild_ch.json \
    --out-file ./prompt_data/chinese_data/instinwild_ch_vicuna.json

python chatllms/data/utils/convert_alpaca.py \
    --in-file ./prompt_data/InstructionWild/instinwild_en.json \
    --out-file ./prompt_data/chinese_data/instinwild_en_vicuna.json
