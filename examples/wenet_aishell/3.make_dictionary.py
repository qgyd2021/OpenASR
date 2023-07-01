#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import librosa
import yaml
import torch
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_filename', required=True, type=str)
    parser.add_argument('--train_data_text', required=True, type=str)
    args = parser.parse_args()
    return args


def text2token(text_file: str, space: str = '<space>'):
    counter = Counter()
    nchar = 1

    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            x = line.split()
            a = ' '.join(x[1:])

            a = [a[j:j + nchar] for j in range(0, len(a), nchar)]

            a_flat = []
            for z in a:
                a_flat.append(''.join(z))
            a_chars = [z.replace(' ', space) for z in a_flat]

            counter.update(a_chars)

    result = list()
    for token, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        result.append(token)
    return result


def main():
    args = get_args()

    tokens = text2token(args.train_data_text)

    dict_filename = Path(args.dict_filename)
    dict_filename.parent.mkdir(parents=True, exist_ok=True)

    with open(args.dict_filename, 'w', encoding='utf-8') as f:
        f.write('<blank> 0\n')
        f.write('<unk> 1\n')

        for idx, token in tqdm(enumerate(tokens)):
            f.write('{} {}\n'.format(token, idx + 2))
        else:
            f.write('<sos/eos> {}\n'.format(idx + 3))

    return


if __name__ == '__main__':
    main()
