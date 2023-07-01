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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wav_file', help='wav file')
    parser.add_argument('--text_file', help='text file')
    parser.add_argument('--output_file', help='output list file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    wav_table = dict()
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            idx, wav_file = line.strip().split(' ', maxsplit=1)
            wav_table[idx] = wav_file

    with open(args.text_file, 'r', encoding='utf8') as fin, \
            open(args.output_file, 'w', encoding='utf8') as fout:
        for line in fin:
            key, txt = line.strip().split(maxsplit=1)

            if key not in wav_table:
                print('key not in wav_table, key: {}'.format(key))
                continue

            wav = wav_table[key]
            row = {
                'key': key,
                'wav': wav,
                'txt': txt,
            }

            json_row = json.dumps(row, ensure_ascii=False)
            fout.write('{}\n'.format(json_row))

    return


if __name__ == '__main__':
    main()
