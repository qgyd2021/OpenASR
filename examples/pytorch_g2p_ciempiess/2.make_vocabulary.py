#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./file_dir', type=str)

    parser.add_argument(
        '--ciempiess_dataset_dir',
        default='D:/programmer/asr_datasets/ciempiess',
        # required=True,
        type=str
    )

    parser.add_argument('--src_dict', default='src_dict.txt', type=str)
    parser.add_argument('--tgt_dict', default='tgt_dict.txt', type=str)

    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='valid.json', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    file_dir = Path(args.file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    ciempiess_dataset_dir = Path(args.ciempiess_dataset_dir)

    # tgt
    phone_file = ciempiess_dataset_dir / 'data/sphinx_experiments/T22_NOTONIC/T22ST_CIEMPIESS.phone'
    with open(file_dir / args.tgt_dict, 'w', encoding='utf-8') as ftgt_dict, \
            open(phone_file, 'r', encoding='utf-8') as fphone:
        ftgt_dict.write('<blank> 0\n')
        ftgt_dict.write('<unk> 1\n')

        idx = 2
        for row in fphone:
            row = str(row).strip()
            ftgt_dict.write('{} {}\n'.format(row, idx))
            idx += 1
        else:
            ftgt_dict.write('<sos/eos> {}\n'.format(idx))

    # src
    with open(file_dir / args.train_subset, 'r', encoding='utf-8') as ftrain:
        counter = Counter()
        for row in ftrain:
            row = json.loads(row)
            src = row['src']
            counter.update(src.split())

    tokens = list()
    for token, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        tokens.append(token)

    with open(file_dir / args.src_dict, 'w', encoding='utf-8') as f:
        f.write('<blank> 0\n')
        f.write('<unk> 1\n')

        for idx, token in tqdm(enumerate(tokens)):
            f.write('{} {}\n'.format(token, idx + 2))
        else:
            f.write('<sos/eos> {}\n'.format(idx + 3))

    return


if __name__ == '__main__':
    main()
