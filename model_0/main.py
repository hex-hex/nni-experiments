import argparse

import nni
from tensorflow import keras
from keras.initializers import RandomUniform, he_uniform
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from model_0 import get_data_set

random_seed = 2020


if __name__ == '__main__':

    data_set = get_data_set('ml100k')
    rating_scaler = MinMaxScaler()
    data_set['rating'] = rating_scaler.fit_transform(data_set['rating'].to_numpy().reshape(-1, 1))
    data_set['user_id'] = data_set['user_id'].astype('category').cat.codes.values
    data_set['item_id'] = data_set['item_id'].astype('category').cat.codes.values
    train, test = train_test_split(data_set, test_size=0.2, random_state=random_seed)
    train, val = train_test_split(train, test_size=0.2, random_state=random_seed)

    embedding_init = RandomUniform(seed=random_seed)
    embeddings_regu = l2(param['regularizer'])

    n_users, n_movies = len(data_set.user_id.unique()), len(data_set.item_id.unique())
    n_latent_factors = param['hidden_factors']
    movie_input = keras.layers.Input(shape=[1], name='Item')
    movie_embedding = keras.layers.Embedding(n_movies, n_latent_factors,
                                             embeddings_initializer=embedding_init,
                                             embeddings_regularizer=embeddings_regu,
                                             name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

    user_input = keras.layers.Input(shape=[1], name='User')
    user_embedding = keras.layers.Embedding(n_users, n_latent_factors,
                                            embeddings_initializer=embedding_init,
                                            embeddings_regularizer=embeddings_regu,
                                            name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
    prod = keras.layers.dot([movie_vec, user_vec], axes=1, normalize=True, name='DotProduct')
    model = keras.Model([user_input, movie_input], prod)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

    history = model.fit([train.user_id, train.item_id], train.rating, epochs=5, verbose=2)



