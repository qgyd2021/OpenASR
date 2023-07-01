#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='data', type=str)
    parser.add_argument('--yesno_cn_dir', default='yesno_cn', type=str)
    parser.add_argument('--metadata', default='train_list.txt', type=str)
    parser.add_argument('--train_subset', default='train.json', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    file_dir = Path(args.file_dir)

    with open(file_dir / args.yesno_cn_dir / args.metadata, 'r', encoding='utf-8') as fin, \
            open(file_dir / args.train_subset, 'w', encoding='utf-8') as fout:
        for row in fin:
            row = str(row).strip()
            wav, txt = row.split(maxsplit=1)
            wav = file_dir / args.yesno_cn_dir / wav

            row = {
                'key': wav.stem,
                'wav': wav.as_posix(),
                'txt': txt,
            }
            row = json.dumps(row, ensure_ascii=False)
            fout.write('{}\n'.format(row))

    return


if __name__ == '__main__':
    main()
