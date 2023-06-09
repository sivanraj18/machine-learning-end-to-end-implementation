import os
import pandas as pd
import yaml
import numpy
from get_data import read_params,get_data
import argparse

def load_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    dataset = config['load_data']['raw_dataset_csv']
    df.to_csv(dataset,index = False)

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config',default = 'params.yaml')
    arg = arg_parse.parse_args()
    load_save(config_path = arg.config)