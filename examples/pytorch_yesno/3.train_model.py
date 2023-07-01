#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import random
import shutil

from pytorch_lightning.trainer.trainer import Trainer
from tqdm import tqdm

from utils.init_model import init_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='file_dir', type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = init_model()
    print(model)
    trainer = Trainer()
    print(trainer)
    return


if __name__ == '__main__':
    main()
