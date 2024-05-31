from modelscope import snapshot_download

if __name__ == '__main__':
    cash_dir = '/home/robin/huggingface_hub/models'
    model_name = 'LLM-Research/Meta-Llama-3-8B-Instruct'
    # model_name = "LLM-Research/Meta-Llama-3-8B"
    # download the model
    model_dir = snapshot_download(model_name,
                                  cache_dir=cash_dir,
                                  revision='master')
