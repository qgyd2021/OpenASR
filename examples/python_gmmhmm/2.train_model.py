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

    # load data
    with open(file_dir / 'target2dataset.pkl', 'rb') as f:
        target2dataset = pickle.load(f)

    # 训练
    target2model: Dict[str, hmm.GMMHMM] = dict()
    for target, dataset in target2dataset.items():
        x_train = dataset['x_train']

        model: hmm.GMMHMM = hmm.GMMHMM(
            n_components=args.m_num_of_hmm_states,
            n_mix=args.m_num_of_mixtures,
            covariance_type=args.m_covariance_type,
            n_iter=args.m_n_iter,
        )
        model.fit(x_train)
        target2model[target] = model

    with open(file_dir / 'target2model.pkl', 'wb') as f:
        pickle.dump(target2model, f)
    return


if __name__ == '__main__':
    main()
