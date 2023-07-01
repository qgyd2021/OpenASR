#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
from typing import Dict, List, Optional

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import torch
import _jsonnet

from toolbox.wenet.models import Model
from toolbox.wenet.common.params import Params
from toolbox.wenet.data.dataset import DatasetReader
from toolbox.wenet.data.vocabulary import Vocabulary
from toolbox.wenet.data.collate_functions import CollateFunction
from toolbox.wenet.training.trainers.trainer import TrainerBase
from toolbox.wenet.training.optimizers import Optimizer
from toolbox.wenet.training.learning_rate_schedulers import LearningRateScheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', default='train_conformer.jsonnet', type=str)

    parser.add_argument('--vocabulary', default='dict.txt', type=str)
    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='dev.json', type=str)

    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epochs', default=240, type=int)
    parser.add_argument('--save_top_k', default=10, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--accumulate_grad_batches', default=4, type=int)
    parser.add_argument('--gradient_clip_val', default=5.0, type=float)
    parser.add_argument('--log_every_n_steps', default=100, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    json_str = _jsonnet.evaluate_file(args.train_config)
    config: dict = json.loads(json_str)
    print(config)

    # model
    model_params = copy.deepcopy(config['model'])
    model = Model.from_params(
        Params(
            params=model_params,
            history='model'
        ),
    )
    # print(model)

    # train data
    dataset_reader_params = copy.deepcopy(config['dataset_reader'])
    train_dataset_reader = DatasetReader.from_params(
        Params(
            params=dataset_reader_params,
            history='train_dataset_reader'
        ),
    )
    train_data_path = config['train_data_path']
    train_dataset_reader.read(train_data_path)

    # valid data
    dataset_reader_params = copy.deepcopy(config['dataset_reader'])
    valid_dataset_reader = DatasetReader.from_params(
        Params(
            params=dataset_reader_params,
            history='valid_dataset_reader'
        ),
    )
    valid_data_path = config['validation_data_path']
    valid_dataset_reader.read(valid_data_path)

    # vocabulary
    vocabulary_params = copy.deepcopy(config['vocabulary'])
    vocabulary = Vocabulary.from_params(
        Params(
            params=vocabulary_params,
            history='vocabulary'
        ),
    )
    # print(vocabulary)

    # train collate_fn
    collate_fn_params = copy.deepcopy(config['collate_fn'])
    train_collate_fn = CollateFunction.from_params(
        Params(
            params=collate_fn_params,
            history='collate_fn'
        ),
    )
    train_collate_fn.train()
    # print(train_collate_fn)

    # valid collate_fn
    collate_fn_params = copy.deepcopy(config['collate_fn'])
    valid_collate_fn = CollateFunction.from_params(
        Params(
            params=collate_fn_params,
            history='collate_fn'
        ),
    )
    valid_collate_fn.train()
    # print(valid_collate_fn)

    # optimizer
    extra = {
        'model_parameters': model.named_parameters(),
    }
    optimizer_params = copy.deepcopy(config['optimizer'])
    optimizer: torch.optim.Optimizer = Optimizer.from_params(
        Params(
            params=optimizer_params,
            history='optimizer'
        ),
        **extra
    )
    # print(optimizer)

    # learning_rate_scheduler
    if 'lr_scheduler' in config.keys():
        extra = {
            'optimizer': optimizer,
        }
        lr_scheduler_params = copy.deepcopy(config['lr_scheduler'])
        lr_scheduler = LearningRateScheduler.from_params(
            Params(
                params=lr_scheduler_params,
                history='lr_scheduler'
            ),
            **extra
        )
    else:
        lr_scheduler = None
    # print(lr_scheduler)

    # trainer
    extra = {
        'model': model,

        'train_dataset': train_dataset_reader,
        'train_collate_fn': train_collate_fn,
        'valid_dataset': valid_dataset_reader,
        'valid_collate_fn': valid_collate_fn,

        'optimizer': optimizer,
        'learning_rate_scheduler': lr_scheduler,

    }
    trainer_params = copy.deepcopy(config['trainer'])
    trainer = TrainerBase.from_params(
        Params(
            params=trainer_params,
            history='trainer'
        ),
        **extra
    )
    # print(trainer)

    trainer.train()
    return


if __name__ == '__main__':
    main()
