#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./file_dir', type=str)

    parser.add_argument(
        '--ciempiess_dataset_dir',
        # default='data_aishell/wav',
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

    # train
    train_transcription_file = ciempiess_dataset_dir / 'data/sphinx_experiments/T22_NOTONIC/T22ST_CIEMPIESS_train.transcription'

    train_key_to_txt = dict()
    with open(train_transcription_file, 'r', encoding='utf-8') as f:
        for row in f:
            split = str(row).strip().split()
            split = [s for s in split if s not in ('<s>', '</s>', '<sil>', '++dis++')]
            txt = ' '.join(split[:-1])
            key = split[-1][1:-1]
            train_key_to_txt[key] = txt

    train_wav_file_list = ciempiess_dataset_dir.glob('data/train/*/*/*.wav')
    with open(file_dir / args.train_subset, 'w', encoding='utf-8') as ftrain:
        for wav_file in tqdm(train_wav_file_list):
            key = wav_file.stem
            txt = train_key_to_txt.get(key)
            if txt is None:
                # print(key)
                continue

            row = {
                'key': key,
                'wav': wav_file.as_posix(),
                'txt': txt
            }
            row = json.dumps(row, ensure_ascii=False)
            ftrain.write('{}\n'.format(row))

    # valid
    valid_transcription_file = ciempiess_dataset_dir / 'data/sphinx_experiments/T22_NOTONIC/T22ST_CIEMPIESS_test.transcription'
    valid_key_to_txt = dict()
    with open(valid_transcription_file, 'r', encoding='utf-8') as f:
        for row in f:
            split = str(row).strip().split()
            split = [s for s in split if s not in ('<s>', '</s>', '<sil>', '++dis++')]
            txt = ' '.join(split[:-1])
            key = split[-1][1:-1]
            valid_key_to_txt[key] = txt

    valid_wav_file_list = ciempiess_dataset_dir.glob('data/test/*/*.wav')
    with open(file_dir / args.valid_subset, 'w', encoding='utf-8') as ftrain:
        for wav_file in tqdm(valid_wav_file_list):
            key = wav_file.stem
            txt = valid_key_to_txt.get(key)
            if txt is None:
                # print(key)
                continue

            row = {
                'key': key,
                'wav': wav_file.as_posix(),
                'txt': txt
            }
            row = json.dumps(row, ensure_ascii=False)
            ftrain.write('{}\n'.format(row))

    return


if __name__ == '__main__':
    main()
