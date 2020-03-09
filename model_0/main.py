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

    model_name = 'model1'
    embedding_init = RandomUniform(seed=random_seed)
    relu_init = he_uniform(seed=random_seed)
    embeddings_regu = l2(1e-6)

    print(val)
