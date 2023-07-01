#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
reference:
https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/

"""
import argparse
import dtw
import numpy as np
from pathlib import Path
from python_speech_features import sigproc
from scipy.io import wavfile

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='file_dir', type=str)
    args = parser.parse_args()
    return args


def wave2spectrum(sample_rate, wave, winlen=0.025, winstep=0.01, nfft=512):
    """计算功率谱图像"""
    frames = sigproc.framesig(
        sig=wave,
        frame_len=winlen * sample_rate,
        frame_step=winstep * sample_rate,
        winfunc=np.hamming
    )
    spectrum = sigproc.powspec(
        frames=frames,
        NFFT=nfft
    )
    spectrum = spectrum.T
    return spectrum


def main():
    args = get_args()
    file_dir = Path(args.file_dir)

    # https://www.ee.columbia.edu/~dpwe/sounds/sents/sm1_cln.wav
    filename1 = file_dir / 'sm1_cln.wav'
    # https://www.ee.columbia.edu/~dpwe/sounds/sents/sm2_cln.wav
    filename2 = file_dir / 'sm2_cln.wav'

    sample_rate1, signal1 = wavfile.read(filename1)
    sample_rate2, signal2 = wavfile.read(filename2)
    max_wave_value = 32768.0
    signal1 = signal1 / max_wave_value
    signal2 = signal2 / max_wave_value

    spectrum1 = wave2spectrum(sample_rate1, signal1)
    spectrum2 = wave2spectrum(sample_rate2, signal2)
    spectrum1 = spectrum1.T
    spectrum2 = spectrum2.T

    cost, C, D1, path = dtw.dtw(spectrum1, spectrum2, dist=lambda a, b: np.sum(a - b) ** 2)

    print(cost)
    # print(C)
    # print(D1)
    # print(path)

    return


if __name__ == '__main__':
    main()
