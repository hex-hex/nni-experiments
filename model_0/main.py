from sklearn.datasets import dump_svmlight_file
import numpy as np
import pandas as pd
import os
import urllib
import zipfile
from sklearn.model_selection import train_test_split
import shutil

# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

datasets = {'ml100k':'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'ml20m':'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
            'mllatestsmall':'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'ml10m':'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
            'ml1m':'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
            }