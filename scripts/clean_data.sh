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


python merge.py \
    --in-file  /userhome/jianzhnie/prompt_data/sharegpt/sharegpt_split.json  /userhome/jianzhnie/prompt_data/sharegpt/evol_instruct_clean.json \
    --out-file /userhome/jianzhnie/prompt_data/sharegpt/evol_sharegpt_merge.json
