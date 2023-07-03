import sys

sys.path.append('../')
from chatllms.data.conv_dataset import VicunaDataset

if __name__ == '__main__':
    data_path = '/home/robin/work_dir/llm/Chinese-Guanaco/examples/sharegpt_formate_role_filter.json'
    # Load the raw data from the specified data_path
    dataset = VicunaDataset(data_path)
    for i, data in enumerate(dataset):
        print(i)
