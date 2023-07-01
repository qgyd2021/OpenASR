#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import sys

import librosa
import torch
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='data', type=str)
    parser.add_argument('--train_subset', default='train.json', type=str)

    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--feat_dim', default=23, type=int)
    parser.add_argument('--frame_length', default=25, type=int)
    parser.add_argument('--frame_shift', default=10, type=int)
    parser.add_argument('--dither', default=0.0, type=int)
    parser.add_argument('--resample_rate', default=8000, type=int)

    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--output_cmvn', default='global_cmvn', type=str)
    parser.add_argument("--log_interval", default=1000, type=int)

    args = parser.parse_args()
    return args


class AudioDataset(Dataset):
    def __init__(self, data_file):
        self.items = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for row in f:
                row = json.loads(row)
                self.items.append(row)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class CollateFunc(object):
    def __init__(self, feat_dim, frame_length, frame_shift, dither, resample_rate):
        self.feat_dim = feat_dim
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.resample_rate = resample_rate

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            wav_file = item['wav']

            waveform, sample_rate = librosa.load(wav_file, sr=self.resample_rate)
            waveform = waveform * (1 << 15)
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(dim=0)

            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.feat_dim,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                sample_frequency=sample_rate
            )
            mean_stat += torch.sum(mat, dim=0)
            var_stat += torch.sum(torch.square(mat), dim=0)
            number += mat.shape[0]
        return number, mean_stat, var_stat


def main():
    args = get_args()
    file_dir = Path(args.file_dir)

    dataset = AudioDataset(file_dir / args.train_subset)
    collate_func = CollateFunc(
        feat_dim=args.feat_dim,
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
        dither=args.dither,
        resample_rate=args.resample_rate,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        collate_fn=collate_func
    )

    with torch.no_grad():
        all_number = 0
        all_mean_stat = torch.zeros(args.feat_dim)
        all_var_stat = torch.zeros(args.feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += args.batch_size

            if wav_number % args.log_interval == 0:
                print('processed {} wavs, {} frames'.format(wav_number, all_number),
                      file=sys.stderr,
                      flush=True)

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }

    with open(file_dir / args.output_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))

    return


if __name__ == '__main__':
    main()
