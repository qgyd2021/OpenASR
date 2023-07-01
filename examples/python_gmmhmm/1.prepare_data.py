#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
from python_speech_features.base import mfcc
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='file_dir', type=str)
    parser.add_argument('--dict_file', default='dict.txt', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    file_dir = Path(args.file_dir)

    filename_list = file_dir.glob('audio/*/*.wav')

    dataset = defaultdict(list)
    labels = list()
    for filename in tqdm(filename_list):
        label = filename.parts[-2]
        sample_rate, signal = wavfile.read(filename)

        feature = mfcc(
            signal=signal,
            samplerate=sample_rate,
            numcep=6,
        )
        dataset[label].append((feature, label))
        if label not in labels:
            labels.append(label)

    # dict.txt
    labels = list(sorted(labels))
    with open(file_dir / args.dict_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write('{}\n'.format(label))

    # pickle
    target2dataset = dict()
    for target, feature_target_list in dataset.items():
        features, targets = zip(*feature_target_list)

        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)

        target2dataset[target] = {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
        }

    with open(file_dir / 'target2dataset.pkl', 'wb') as f:
        pickle.dump(target2dataset, f)
    return


if __name__ == '__main__':
    main()
