import yaml
import argparse
import os
import pandas as pd

def read_params(config_path):
    with open('params.yaml','r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path = config['data_source']['s3_source']
    df = pd.read_csv(data_path,sep = ',')
    return df



if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config',default ='params.yaml' )
    parse = arg_parse.parse_args()
    get_data(config_path = parse.config)

