import os
import zipfile

import requests
import pandas as pd

data_sets = {
    'ml100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
    'mllatestsmall': 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
    'ml10m': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
    'ml1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
}


def download_data(data_path: str = '../source_data/'):
    for k, v in data_sets.items():
        dt_name = os.path.basename(v)
        dt_path = '{}{}'.format(data_path, dt_name)
        if not os.path.exists(dt_path):
            resp = requests.get(v)
            with open(dt_path, 'wb') as f:
                f.write(resp.content)
        if not os.path.exists(os.path.splitext(dt_path)[0]):
            with zipfile.ZipFile(dt_path, 'r') as zip_f:
                zip_f.extractall(data_path)


def get_data_set(data_set_name: str, data_path: str = './source_data/'):
    set_path = '{}{}'.format(data_path, os.path.splitext(os.path.basename(data_sets[data_set_name]))[0])
    if data_set_name == 'ml100k':
        return pd.read_csv(set_path + "/u.data", sep='\t', names="user_id,item_id,rating,timestamp".split(","))

    # ml1m
    if data_set_name == 'ml1m':
        return pd.read_csv(set_path + '/ratings.dat', delimiter='\:\:',
                           names=['user_id', 'item_id', 'rating', 'timestamp'])
