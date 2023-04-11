import os
import argparse
import yaml
import pandas
from sklearn.model_selection import train_test_split
from get_data import read_params


def split_data(config_path):
    config = read_params(config_path)
    dataset = config['load_data']['raw_dataset_csv']
    train_data = config['split_data']['train_path']
    test_data = config['split_data']['test_path']
    random_state = config['base']['random_state']
    plit_ratio = config['split_data']['test_size']

    df = pd.read_csv(dataset,sep = ',')
    


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config',default = 'params.yaml')
    agr = arg_parse.parse_args()
    split_data(config_path=arg.config)