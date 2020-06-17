import argparse
import math
import random
import time

import keras
import pandas as pd
import numpy as np

import nni
import scipy
from keras.initializers import RandomUniform, he_uniform
from keras.regularizers import l2
from keras.utils import Sequence
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

model_name = 'model1'
seed = 2020
embedding_init = RandomUniform(seed=seed)
relu_init = he_uniform(seed=seed)

dataset = pd.read_csv("../source_data/ml-100k/u.data", sep='\t', names="user_id,item_id,rating,timestamp".split(","))
# dataset = pd.read_csv('../source_data/ml-1m/ratings.dat', delimiter='::',
#                       names=['user_id', 'item_id', 'rating', 'timestamp'])
dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
dataset.rating = pd.to_numeric(dataset.rating, errors='coerce')


def neg_sampling(ratings_df, n_neg=1, neg_val=0, pos_val=1, percent_print=5):
    sparse_mat = scipy.sparse.coo_matrix((ratings_df.rating, (ratings_df.user_id, ratings_df.item_id)))
    dense_mat = np.asarray(sparse_mat.todense())
    print(dense_mat.shape)

    nsamples = ratings_df[['user_id', 'item_id']]
    nsamples['rating'] = nsamples.apply(lambda row: 1, axis=1)
    length = dense_mat.shape[0]
    printpc = int(length * percent_print / 100)

    nTempData = []
    i = 0
    start_time = time.time()
    stop_time = time.time()

    extra_samples = 0
    for row in dense_mat:
        if (i % printpc == 0):
            stop_time = time.time()
            print("processed ... {0:0.2f}% ...{1:0.2f}secs".format(float(i) * 100 / length, stop_time - start_time))
            start_time = stop_time

        n_non_0 = len(np.nonzero(row)[0])
        zero_indices = np.where(row == 0)[0]
        if (n_non_0 * n_neg + extra_samples > len(zero_indices)):
            print(i, "non 0:", n_non_0, ": len ", len(zero_indices))
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


class DataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=16, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.shuffle = shuffle
        self.indices = dataframe.index
        print(len(self.indices), "\n", dataframe.isna().any())
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.floor(len(self.dataframe) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]

        # Find list of batch IDs
        list_IDs_temp = [self.indices[k] for k in idxs]

        # Generate data
        User = self.dataframe.iloc[list_IDs_temp, [0]].to_numpy().reshape(-1)
        Item = self.dataframe.iloc[list_IDs_temp, [1]].to_numpy().reshape(-1)
        rating = self.dataframe.iloc[list_IDs_temp, [2]].to_numpy().reshape(-1)
        assert (np.isnan(User).any(), User)
        assert (np.isnan(Item).any(), Item)
        assert (np.isnan(rating).any(), rating)
        # print(idxs, '\n', "u,i,", [User, Item],'\n',[rating])
        return [User, Item], [rating]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


def create_model(dataset, embeddings_regu, n_latent_factors=16):
    n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())

    movie_input = keras.layers.Input(shape=[1], name='Item')
    movie_embedding = keras.layers.Embedding(n_movies, n_latent_factors,
                                             embeddings_initializer=embedding_init,
                                             embeddings_regularizer=embeddings_regu,
                                             embeddings_constraint="NonNeg",
                                             name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

    user_input = keras.layers.Input(shape=[1], name='User')
    user_embedding = keras.layers.Embedding(n_users, n_latent_factors,
                                            embeddings_initializer=embedding_init,
                                            embeddings_regularizer=embeddings_regu,
                                            embeddings_constraint="NonNeg",
                                            name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)

    prod = keras.layers.dot([movie_vec, user_vec], axes=1, name='DotProduct')
    model = keras.Model([user_input, movie_input], prod)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.summary()
    return model


neg_dataset = neg_sampling(dataset)

train, test = train_test_split(neg_dataset, test_size=0.2, random_state=2020)
train, val = train_test_split(train, test_size=0.2, random_state=2020)


def main(param):
    print(param)
    embeddings_regu = l2(param['regularizer'])
    model = create_model(neg_dataset, embeddings_regu, param['hidden_factors'])
    train_generator = DataGenerator(train, batch_size=param['batch'], shuffle=False)
    history = model.fit(train_generator, epochs=50, verbose=2)
    results = model.evaluate([test.user_id, test.item_id], test.rating, batch_size=1, verbose=0)
    nni.report_final_result(results[1])


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_factors", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--regularizer", type=float, default=1e-4)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(get_params())
    params.update(tuner_params)
    main(params)
