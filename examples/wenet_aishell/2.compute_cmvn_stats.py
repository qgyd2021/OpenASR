#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import sys

import librosa
import yaml
import torch
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='extract CMVN stats')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
    parser.add_argument('--train_config',
                        default='',
                        help='training yaml conf')
    parser.add_argument('--in_scp', default=None, help='wav scp file')
    parser.add_argument('--out_cmvn',
                        default='global_cmvn',
                        help='global cmvn file')

    doc = "Print log after every log_interval audios are processed."
    parser.add_argument("--log_interval", type=int, default=1000, help=doc)
    args = parser.parse_args()
    return args


class AudioDataset(Dataset):
    def __init__(self, data_file):
        self.items = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split()
                self.items.append((arr[0], arr[1]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class CollateFunc(object):
    def __init__(self, feat_dim, resample_rate):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            waveform, sample_rate = librosa.load(item[1], sr=self.resample_rate)
            waveform = waveform * (1 << 15)
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(dim=0)

            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.feat_dim,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=sample_rate
            )
            mean_stat += torch.sum(mat, dim=0)
            var_stat += torch.sum(torch.square(mat), dim=0)
            number += mat.shape[0]
        return number, mean_stat, var_stat


def main():
    args = get_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feat_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    resample_rate = 0
    if 'resample_conf' in configs['dataset_conf']:
        resample_rate = configs['dataset_conf']['resample_conf']['resample_rate']
        print('using resample and new sample rate is {}'.format(resample_rate))

    collate_func = CollateFunc(feat_dim, resample_rate)
    dataset = AudioDataset(args.in_scp)
    batch_size = 20
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        collate_fn=collate_func
    )

    with torch.no_grad():
        all_number = 0
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += batch_size

            if wav_number % args.log_interval == 0:
                print('processed {} wavs, {} frames'.format(wav_number, all_number),
                      file=sys.stderr,
                      flush=True)

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }

    with open(args.out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))

    return


if __name__ == '__main__':
    main()
