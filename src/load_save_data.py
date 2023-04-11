import os
import pandas as pd
import yaml
import numpy
from get_data import read_params,get_data
import argparse
from sklearn.model_selection import train_test_split

def load_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    dataset = config['load_data']['raw_dataset_csv']
    train_data = config['split_data']['train_path']
    test_data = config['split_data']['test_path']
    rndm_st = config['base']['random_state']
    size = config['split_data']['test_size']

    train_df,test_df = train_test_split(df,random_state=rndm_st,test_size=size)

    train_df.to_csv(train_data,index = False)
    test_df.to_csv(test_data,index = False)
    df.to_csv(dataset,index = False)

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config',default = 'params.yaml')
    arg = arg_parse.parse_args()
    load_save(config_path = arg.config)