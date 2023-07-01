#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pickle

from typing import Dict

from hmmlearn import hmm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='file_dir', type=str)
    parser.add_argument('--dict_file', default='dict.txt', type=str)

    parser.add_argument('--m_num_of_hmm_states', default=3, type=int)
    parser.add_argument('--m_num_of_mixtures', default=2, type=int)
    parser.add_argument('--m_covariance_type', default='diag', type=str)
    parser.add_argument('--m_n_iter', default=10, type=int)
    parser.add_argument('--m_bakis_level', default=2, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    file_dir = Path(args.file_dir)

    # load dict
    token2index = dict()
    with open(file_dir / args.dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            label = str(line).strip()
            token2index[label] = len(token2index)
    index2token = {v: k for k, v in token2index.items()}

    # load data
    with open(file_dir / 'target2dataset.pkl', 'rb') as f:
        target2dataset = pickle.load(f)

    # load models
    with open(file_dir / 'target2model.pkl', 'rb') as f:
        target2model: Dict[str, hmm.GMMHMM] = pickle.load(f)

    # predict train subset
    for target, dataset in target2dataset.items():
        x_train = dataset['x_train']
        # x_test = dataset['x_test']
        y_train = dataset['y_train']
        # y_test = dataset['y_test']

        for x, y in zip(x_train, y_train):
            scores = list()
            for _, model in target2model.items():
                score = model.score([x])
                scores.append(score)

            idx = scores.index(max(scores))
            pred = index2token[idx]
            print('target: {}, predict: {}'.format(y, pred))

    # predict test subset
    for target, dataset in target2dataset.items():
        # x_train = dataset['x_train']
        x_test = dataset['x_test']
        # y_train = dataset['y_train']
        y_test = dataset['y_test']

        for x, y in zip(x_test, y_test):
            scores = list()
            for _, model in target2model.items():
                score = model.score([x])
                scores.append(score)

            idx = scores.index(max(scores))
            pred = index2token[idx]
            print('target: {}, predict: {}'.format(y, pred))

    return


if __name__ == '__main__':
    main()
