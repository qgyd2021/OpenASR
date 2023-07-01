#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aishell_audio_dir',
        # default='data_aishell/wav',
        default='D:/programmer/asr_datasets/aishell/data_aishell/wav',
        # required=True,
        type=str
    )
    parser.add_argument(
        '--aishell_text',
        # default='data_aishell/transcript/aishell_transcript_v0.8.txt',
        default='D:/programmer/asr_datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt',
        # required=True,
        type=str
    )

    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='dev.json', type=str)
    parser.add_argument('--test_subset', default='test.json', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    key_to_txt = dict()
    with open(args.aishell_text, 'r', encoding='utf-8') as f:
        for row in f:
            key, txt = str(row).strip().split(maxsplit=1)
            txt = ''.join(txt.split())
            key_to_txt[key] = txt

    aishell_audio_dir = Path(args.aishell_audio_dir)
    wav_file_list = aishell_audio_dir.glob('**/*.wav')

    with open(args.train_subset, 'w', encoding='utf-8') as ftrain, \
        open(args.test_subset, 'w', encoding='utf-8') as ftest, \
            open(args.valid_subset, 'w', encoding='utf-8') as fdev:
        for wav_file in tqdm(wav_file_list):
            key = wav_file.stem
            split = wav_file.parts[-3]

            txt = key_to_txt.get(key)
            if txt is None:
                continue

            row = {
                'key': key,
                'wav': wav_file.as_posix(),
                'txt': txt,
            }
            row = json.dumps(row, ensure_ascii=False)
            if split == 'train':
                ftrain.write('{}\n'.format(row))
            elif split == 'test':
                ftest.write('{}\n'.format(row))
            elif split == 'dev':
                fdev.write('{}\n'.format(row))
            else:
                raise AssertionError
    return


if __name__ == '__main__':
    main()
