#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
reference:
https://github.com/wenet-e2e/wenet/blob/main/wenet/bin/recognize.py

"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Optional

pwd = os.path.abspath(os.path.dirname(__file__))
project_path = Path(os.path.join(pwd, '../../'))
sys.path.append(project_path.as_posix())

import _wenet
import librosa
import yaml
import torch
from tqdm import tqdm

# wenetruntime 只能在 linux 上使用. pip3 install wenetruntime
# import wenetruntime as wenet


def get_args():
    """
    python3 recognize_by_filename.py --filename voicemail.wav --pretrained_model /data/tianxing/PycharmProjects/OpenASR/pretrained_models/wenet/20220506_u2pp_conformer_exp
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='file_dir', type=str)
    parser.add_argument(
        '--pretrained_model',
        default='D:/programmer/asr_pretrained_model/wenet/wenetspeech_u2pp_conformer_exp/20220506_u2pp_conformer_exp',
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    decoder = wenet.Decoder(
        model_dir=args.pretrained_model
    )

    signal, sample_rate = librosa.load(args.filename)
    ans = decoder.decode(signal.tobytes(), True)
    print(ans)
    return


if __name__ == '__main__':
    main()
