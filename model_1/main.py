import argparse

import nni


def main(param):
    print(param)


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
