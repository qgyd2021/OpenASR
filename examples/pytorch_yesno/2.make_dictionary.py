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
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    file_dir = Path(args.file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    with open(file_dir / 'dict.txt', 'w', encoding='utf-8') as f:
        f.write('<blank> 0\n')
        f.write('<unk> 1\n')
        f.write('zero 2\n')
        f.write('one 3\n')
        f.write('<sos/eos> 4\n')

    return


if __name__ == '__main__':
    main()
