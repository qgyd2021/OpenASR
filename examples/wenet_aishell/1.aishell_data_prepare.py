#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考文件:
thirdparty/wenet-2.1.0/examples/aishell/s0/local/aishell_data_prep.sh

用 python 来实现这个脚本. 之所以这样做, 是为了统一用 python 脚本来做数据预处理过程.

/c/Users/tianx/PycharmProjects/virtualenv/OpenASR/Scripts/python.exe 1.aishell_data_prepare.py \
--aishell_audio_dir D:/programmer/asr_datasets/aishell/data_aishell/wav \
--aishell_text D:/programmer/asr_datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt

python3 1.aishell_data_prepare.py \
--aishell_audio_dir D:/programmer/asr_datasets/aishell/data_aishell/wav \
--aishell_text D:/programmer/asr_datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt

备注:
sort -u 排序后删除重复行

"""
import argparse
from pathlib import Path
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', type=str)

    parser.add_argument('--aishell_audio_dir', default='data_aishell/wav', required=True, type=str)
    parser.add_argument('--aishell_text', default='data_aishell/transcript/aishell_transcript_v0.8.txt',
                        required=True, type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / 'local/train'
    dev_dir = data_dir / 'local/dev'
    test_dir = data_dir / 'local/test'
    tmp_dir = data_dir / 'local/tmp'

    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    aishell_audio_dir = Path(args.aishell_audio_dir)
    wav_file_list = aishell_audio_dir.glob('**/*.wav')

    count = 0
    with open(train_dir / 'wav.flist', 'w', encoding='utf-8') as ftrain, \
            open(dev_dir / 'wav.flist', 'w', encoding='utf-8') as fdev, \
            open(test_dir / 'wav.flist', 'w', encoding='utf-8') as ftest, \
            open(tmp_dir / 'wav.flist', 'w', encoding='utf-8') as ftmp:
        for wav_file in wav_file_list:
            if str(wav_file).__contains__('train'):
                ftrain.write('{}\n'.format(wav_file))
            if str(wav_file).__contains__('dev'):
                fdev.write('{}\n'.format(wav_file))
            if str(wav_file).__contains__('test'):
                ftest.write('{}\n'.format(wav_file))
            ftmp.write('{}\n'.format(wav_file))

            count += 1

    if count != 141925:
        print('Warning: expected 141925 data data files, found {}'.format(count))

    shutil.rmtree(tmp_dir)

    # Transcriptions preparation
    for directory in [train_dir, dev_dir, test_dir]:
        print('Preparing {} transcriptions'.format(directory.as_posix()))

        utt_list = set()
        utt2wav = dict()
        with open(directory / 'wav.flist', 'r', encoding='utf-8') as fin, \
                open(directory / 'utt.list', 'w', encoding='utf-8') as futt, \
                open(directory / 'wav.scp_all', 'w', encoding='utf-8') as fscp_all:
            for row in fin:
                filename = Path(str(row).strip())
                utt = filename.stem

                futt.write('{}\n'.format(utt))
                fscp_all.write('{} {}\n'.format(utt, filename))
                utt_list.add(utt)
                utt2wav[utt] = filename.as_posix()

        utt2text = dict()
        with open(args.aishell_text, 'r', encoding='utf-8') as ftext:
            for row in ftext:
                utt, text = str(row).strip().split(' ', maxsplit=1)
                utt2text[utt] = text

        with open(directory / 'transcripts.txt', 'w', encoding='utf-8') as ftranscripts, \
                open(directory / 'utt.list', 'w', encoding='utf-8') as futt:
            for utt in utt_list:
                if utt not in utt2text:
                    print('utt: {} not in transcripts.txt'.format(utt))
                    continue
                text = utt2text[utt]
                ftranscripts.write('{} {}\n'.format(utt, text))
                futt.write('{}\n'.format(utt))

        with open(directory / 'wav.scp', 'w', encoding='utf-8') as fscp:
            rows = list()
            for utt in utt_list:
                if utt not in utt2wav:
                    continue
                wav_file = utt2wav[utt]
                rows.append('{} {}'.format(utt, wav_file))

            for row in sorted(rows):
                fscp.write('{}\n'.format(row))

        with open(directory / 'transcripts.txt', 'r', encoding='utf-8') as ftranscripts, \
                open(directory / 'text', 'w', encoding='utf-8') as ftext:
            rows = list()
            for row in ftranscripts:
                idx, text = row.strip().split(' ', maxsplit=1)
                text = ''.join(text.split(' '))
                row = '{} {}'.format(idx, text)
                rows.append(row)

            for row in sorted(rows):
                ftext.write('{}\n'.format(row))

    train_dir_ = data_dir / 'train'
    dev_dir_ = data_dir / 'dev'
    test_dir_ = data_dir / 'test'

    train_dir_.mkdir(parents=True, exist_ok=True)
    dev_dir_.mkdir(parents=True, exist_ok=True)
    test_dir_.mkdir(parents=True, exist_ok=True)

    for filename in ['wav.scp', 'text']:
        shutil.copy(train_dir / filename, train_dir_ / filename)
        shutil.copy(dev_dir / filename, dev_dir_ / filename)
        shutil.copy(test_dir / filename, test_dir_ / filename)

    return


if __name__ == '__main__':
    main()
