#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_asr_data_dir', default='./file_dir/LibriSpeech/train-clean-5', type=str)
    parser.add_argument('--valid_asr_data_dir', default='./file_dir/LibriSpeech/dev-clean-2', type=str)

    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='valid.json', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # train
    train_asr_data_dir = Path(args.train_asr_data_dir)
    key_to_txt = dict()
    for transcription in tqdm(train_asr_data_dir.glob('*/*/*.txt')):
        with open(transcription, 'r', encoding='utf-8') as f:
            for row in f:
                key, txt = str(row).strip().split(maxsplit=1)
                key_to_txt[key.lower()] = txt.lower()

    with open(args.train_subset, 'w', encoding='utf-8') as f:
        for wav_file in tqdm(train_asr_data_dir.glob('*/*/*.flac')):
            key = wav_file.stem
            txt = key_to_txt.get(key)
            if txt is None:
                print('no txt for wav: {}'.format(wav_file))
                continue

            row = {
                'key': key,
                'wav': wav_file.as_posix(),
                'txt': txt,
            }
            row = json.dumps(row, ensure_ascii=False)
            f.write('{}\n'.format(row))

    # valid
    valid_asr_data_dir = Path(args.valid_asr_data_dir)
    key_to_txt = dict()
    for transcription in tqdm(valid_asr_data_dir.glob('*/*/*.txt')):
        with open(transcription, 'r', encoding='utf-8') as f:
            for row in f:
                key, txt = str(row).strip().split(maxsplit=1)
                key_to_txt[key.lower()] = txt.lower()

    with open(args.valid_subset, 'w', encoding='utf-8') as f:
        for wav_file in tqdm(valid_asr_data_dir.glob('*/*/*.flac')):
            key = wav_file.stem
            txt = key_to_txt.get(key)
            if txt is None:
                print('no txt for wav: {}'.format(wav_file))
                continue

            row = {
                'key': key,
                'wav': wav_file.as_posix(),
                'txt': txt,
            }
            row = json.dumps(row, ensure_ascii=False)
            f.write('{}\n'.format(row))

    return


if __name__ == '__main__':
    main()
