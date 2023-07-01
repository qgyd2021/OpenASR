#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
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

    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='valid.json', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    file_dir = Path(args.file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    ciempiess_dataset_dir = Path(args.ciempiess_dataset_dir)

    # pronunciation dictionary
    dic_file = ciempiess_dataset_dir / 'data/sphinx_experiments/T22_NOTONIC/T22ST_CIEMPIESS.dic'
    with open(dic_file, 'r', encoding='utf-8') as fdic, \
            open(file_dir / args.train_subset, 'w', encoding='utf-8') as ftrain, \
            open(file_dir / args.valid_subset, 'w', encoding='utf-8') as fvalid:
        for row in fdic:
            row = str(row).strip()
            if row.__contains__('<UNK>'):
                continue
            splits = row.split()
            src = ' '.join(list(splits[0]))
            tgt = ' '.join(splits[1:])

            row = {
                'src': src,
                'tgt': tgt
            }
            row = json.dumps(row, ensure_ascii=False)

            if random.random() < 0.9:
                ftrain.write('{}\n'.format(row))
            else:
                fvalid.write('{}\n'.format(row))
    return


if __name__ == '__main__':
    main()
