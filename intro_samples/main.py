import os
import nni

from intro_samples import input_data


def main():
    pass


if __name__ == '__main__':
    params = {
        'data_dir': './sample_data/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32,
        'channel_2_num': 64,
        'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000,
        'batch_size': 32
    }
    input_data.read_data_sets('./input_data', one_hot=True)
    main()
