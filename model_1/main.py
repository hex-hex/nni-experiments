import argparse
import random
import time

import pandas as pd
import numpy as np

import nni
import scipy
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split


def neg_sampling(ratings_df, n_neg=1, neg_val=0, pos_val=1, percent_print=5):
    """version 1.2: 1 positive 1 neg (2 times bigger than the original dataset by default)

      Parameters:
      input rating data as pandas dataframe: userId|movieId|rating
      n_neg: include n_negative / 1 positive

      Returns:
      negative sampled set as pandas dataframe
              userId|movieId|interact (implicit)
    """
    sparse_mat = coo_matrix((ratings_df.rating, (ratings_df.user_id, ratings_df.item_id)), dtype=np.float64)
    dense_mat = np.asarray(sparse_mat.todense())
    print(dense_mat.shape)

    nsamples = ratings_df[['user_id', 'item_id']]
    nsamples.loc['rating'] = nsamples.apply(lambda row: 1, axis=1)
    length = dense_mat.shape[0]
    printpc = int(length * percent_print / 100)

    nTempData = []
    i = 0
    # start_time = time.time()
    # stop_time = time.time()

    extra_samples = 0
    for row in dense_mat:
        if i % printpc == 0:
            stop_time = time.time()
            # print("processed ... {0:0.2f}% ...{1:0.2f}secs".format(float(i) * 100 / length, stop_time - start_time))
            start_time = stop_time

        n_non_0 = len(np.nonzero(row)[0])
        zero_indices = np.where(row == 0)[0]
        if n_non_0 * n_neg + extra_samples > len(zero_indices):
            # print(i, "non 0:", n_non_0, ": len ", len(zero_indices))
            neg_indices = zero_indices.tolist()
            extra_samples = n_non_0 * n_neg + extra_samples - len(zero_indices)
        else:
            neg_indices = random.sample(zero_indices.tolist(), n_non_0 * n_neg + extra_samples)
            extra_samples = 0

        nTempData.extend([(uu, ii, rr) for (uu, ii, rr) in zip(np.repeat(i, len(neg_indices))
                                                               , neg_indices, np.repeat(neg_val, len(neg_indices)))])
        i += 1

    nsamples = nsamples.append(pd.DataFrame(nTempData, columns=["user_id", "item_id", "rating"]), ignore_index=True)
    nsamples.reset_index(drop=True)
    return nsamples


dataset = pd.read_csv('../source_data/ml-latest-small/ratings.csv', usecols=[0, 1, 2, 3],
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
dataset.rating = pd.to_numeric(dataset.rating, errors='coerce')

neg_dataset = neg_sampling(dataset)

train, test = train_test_split(neg_dataset, test_size=0.2, random_state=2020)
train, val = train_test_split(train, test_size=0.2, random_state=2020)


def main(param):
    print(param)
    print(train)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_factors", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--regularizer", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(get_params())
    params.update(tuner_params)
    main(params)
