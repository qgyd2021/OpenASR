#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import random
import shutil

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='file_dir', type=str)
    parser.add_argument('--yesno_dir', default='waves_yesno', required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    file_dir = Path(args.file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    yesno_dir = Path(args.yesno_dir)

    with open(file_dir / 'train.json', 'w', encoding='utf-8') as ftrain, \
            open(file_dir / 'valid.json', 'w', encoding='utf-8') as ftest:
        for filename in tqdm(yesno_dir.glob('*.wav')):
            tokens = ['zero' if token == '0' else 'one' for token in filename.stem.split('_')]
            txt = ' '.join(tokens)

            row = {
                'key': filename.stem,
                'wav': filename.as_posix(),
                'txt': txt
            }
            json_row = json.dumps(row)

            if random.random() < 0.9:
                ftrain.write('{}\n'.format(json_row))
            else:
                ftest.write('{}\n'.format(json_row))

    return


if __name__ == '__main__':
    main()
