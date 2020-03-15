import argparse
import random
import time

import keras
import pandas as pd
import numpy as np

import nni
import tensorflow as tf
from keras import Input, Model, layers
from keras.initializers import RandomUniform, he_uniform
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

model_name = 'model1'
seed = 2020
embedding_init = RandomUniform(seed=seed)
relu_init = he_uniform(seed=seed)

dataset = pd.read_csv('../source_data/ml-latest-small/ratings.csv', usecols=[0, 1, 2, 3],
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
dataset.rating = pd.to_numeric(dataset.rating, errors='coerce')

uids = np.sort(dataset.user_id.unique())
iids = np.sort(dataset.item_id.unique())

n_users = len(uids)
n_items = len(iids)


def neg_sampling(ratings_df, n_neg=1, neg_val=0, pos_val=1, percent_print=5):
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


def create_rating_matrix(u_i_r_df):
    rating_matrix = np.zeros(shape=(n_users, n_items), dtype=int)
    for row in u_i_r_df.itertuples(index=False):
        rating_matrix[int(row[0]), int(row[1])] = int(row[2])
    return rating_matrix


neg_dataset = neg_sampling(dataset)

train, test = train_test_split(neg_dataset, test_size=0.2, random_state=2020)
train, val = train_test_split(train, test_size=0.2, random_state=2020)


def create_hidden_size(n_hidden_layers=3, n_latent_factors=32):
    hidden_size = [n_latent_factors * 2 ** i for i in reversed(range(n_hidden_layers))]
    return hidden_size


def create_model(n_users, n_items, learning_rate, n_hidden_layers, n_latent_factors, l1=1e-5, l2=1e-4):
    # hidden_size = create_hidden_size()
    hidden_size = create_hidden_size(n_hidden_layers, n_latent_factors)

    # create 4 input layers
    uii = Input(shape=(n_items,), name='uii')
    umi = Input(shape=(n_items,), name='umi')  # is a neighbour of ui

    vji = Input(shape=(n_users,), name='vji')
    vni = Input(shape=(n_users,), name='vni')  # is a neighour of vj

    # user autoencoder
    encoded = uii
    for nn in hidden_size[:-1]:
        encoded = Dense(nn, activation='relu',
                        kernel_initializer='he_uniform',
                        # kernel_regularizer=l2(l2_val)
                        )(encoded)
        # encoded = BatchNormalization()(encoded)
        # encoded = Dropout(0.2)(encoded)

    encoded = Dense(hidden_size[-1], activation='relu',
                    kernel_initializer='he_uniform',
                    # kernel_regularizer=l2(l2_val),
                    name='encoder')(encoded)

    hidden_size.reverse()
    decoded = encoded
    for nn in hidden_size[1:]:
        decoded = Dense(nn, activation='relu',
                        kernel_initializer='he_uniform',
                        # kernel_regularizer=l2(l2_val)
                        )(decoded)
        # decoded = BatchNormalization()(decoded)
        # decoded = Dropout(0.2)(decoded)
    decoded = Dense(n_items, activation='relu',
                    kernel_initializer='he_uniform',
                    # kernel_regularizer=l2(l2_val),
                    name='decoder')(decoded)

    # for item autoencoder
    # hidden_size = create_hidden_size() #reset hidden size
    hidden_size = create_hidden_size(n_hidden_layers, n_latent_factors)  # reset hidden size
    encoded2 = vji
    for nn in hidden_size[:-1]:
        encoded2 = Dense(nn, activation='relu',
                         kernel_initializer='he_uniform',
                         #  kernel_regularizer=l2(l2_val)
                         )(encoded2)
        # encoded2 = BatchNormalization()(encoded2)
        # encoded2 = Dropout(0.2)(encoded2)

    encoded2 = Dense(hidden_size[-1], activation='relu',
                     kernel_initializer='he_uniform',
                     #  kernel_regularizer=l2(l2_val),
                     name='encoder2')(encoded2)

    hidden_size.reverse()
    decoded2 = encoded2
    for nn in hidden_size[1:]:
        decoded2 = Dense(nn, activation='relu',
                         kernel_initializer='he_uniform',
                         #  kernel_regularizer=l2(l2_val)
                         )(decoded2)
        # decoded2 = BatchNormalization()(decoded2)
        # decoded2 = Dropout(0.2)(decoded2)

    decoded2 = Dense(n_users, activation='relu',
                     kernel_initializer='he_uniform',
                     # kernel_regularizer=l2(l2_val),
                     name='decoder2')(decoded2)

    # prod = layers.dot([encoded, encoded2], axes=1, name='DotProduct')
    # V2: replace dot prod with mlp
    concat = layers.concatenate([encoded, encoded2])
    mlp = concat
    for i in range(3, -1, -1):
        if i == 0:
            mlp = Dense(1, activation='sigmoid',
                        name="output")(mlp)
        else:
            mlp = Dense(8 * 2 ** i, activation='sigmoid',
                        # kernel_regularizer=l1_l2(l1_val,l2_val),
                        # kernel_initializer=ß'he_uniform'
                        )(mlp)
            if i >= 2:
                mlp = BatchNormalization()(mlp)
                mlp = Dropout(0.2)(mlp)

    model = Model(inputs=[uii, vji], outputs=[decoded, decoded2, mlp])
    adadelta = tf.keras.optimizers.Adadelta(learning_rate)
    model.compile(optimizer='adadelta', loss={'output': 'binary_crossentropy',
                                              'decoder': 'mean_squared_error',
                                              'decoder2': 'mean_squared_error'
                                              },
                  metrics={'output': ['binary_accuracy',
                                      #  'Precision', 'AUC'
                                      ],
                           'decoder': 'mse',
                           'decoder2': 'mse'})

    # model.summary()

    return model


def main(param):
    print(param)
    model = create_model(neg_dataset, param['hidden_factors'], param['lr'], param['regularizer'])
    history = model.fit([train.user_id, train.item_id], train.rating, batch_size=param['batch'],
                        epochs=10, verbose=0)
    results = model.evaluate([test.user_id, test.item_id], test.rating, batch_size=1, verbose=0)
    nni.report_final_result(results[1])


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_factors", type=int, default=8)
    parser.add_argument("--hidden_layers", type=int, default=3)
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
