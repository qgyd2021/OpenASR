#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import json
from pathlib import Path

from tqdm import tqdm

from toolbox.wenet.utils.data.tokenizers.asr_tokenizer import CjkBpeTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='data', type=str)
    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--vocabulary', default='dict.txt', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    file_dir = Path(args.file_dir)

    tokenizer = CjkBpeTokenizer()
    counter = Counter()

    with open(file_dir / args.train_subset, 'r', encoding='utf-8') as fin:
        for row in fin:
            row = json.loads(row)
            txt = row['txt']
            tokens = tokenizer.tokenize(txt)
            counter.update(tokens)

    tokens = list()
    for token, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        tokens.append(token)

    with open(file_dir / args.vocabulary, 'w', encoding='utf-8') as f:
        f.write('<blank> 0\n')
        f.write('<unk> 1\n')

        for idx, token in tqdm(enumerate(tokens)):
            f.write('{} {}\n'.format(token, idx + 2))
        else:
            f.write('<sos/eos> {}\n'.format(idx + 3))

    return


if __name__ == '__main__':
    main()
