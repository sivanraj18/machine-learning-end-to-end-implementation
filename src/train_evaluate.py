import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json
from get_data import read_params
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train = train.dropna()
    test = test.dropna()
    
    train_y = train[target]
    test_y = test[target]


    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)


    cat_col = [col for col in train_x.columns if train_x[col].dtype == 'O']
    num_col = [col for col in train_x.columns if train_x[col].dtype != 'O' and col not in target]

    cat_pipe = Pipeline([
        ('simple_imputer',SimpleImputer(strategy='most_frequent')),
        ('onehotencoding',OneHotEncoder())
        ])
    
    num_pipe  = Pipeline([
        ('simple_imputer',SimpleImputer(strategy='mean')),
        ('standard_scaler',StandardScaler())
        ])


    ct = ColumnTransformer([('for_categorical_values',cat_pipe,cat_col),
                            ('for_num_vslues',num_pipe,num_col)
                            ])

    
    train_x = pd.DataFrame(ct.fit_transform(train_x))
    test_x = pd.DataFrame(ct.transform(test_x))

    lr = ElasticNet(
        alpha=alpha, 
        l1_ratio=l1_ratio, 
        random_state=random_state)
    lr.fit(train_x,train_y)

    predicted_qualities = lr.predict(test_x)
    
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2) 
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        json.dump(params, f, indent=4)
#####################################################


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)




if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config',default = 'params.yaml')
    agr = arg_parse.parse_args()
    train_and_evaluate(config_path = agr.config)