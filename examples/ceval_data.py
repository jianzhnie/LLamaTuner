from datasets import load_dataset

dataset = load_dataset("/home/robin/prompt_data/ceval-exam",
                       "computer_network")

print(dataset)
