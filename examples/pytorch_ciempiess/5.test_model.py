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

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import _jsonnet

from toolbox.wenet.data.collate_function import CollateFunction
from toolbox.wenet.data.dataset.dataset import DatasetReader
from toolbox.wenet.data.preprocess import Preprocess
from toolbox.wenet.models.model import Model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./file_dir', type=str)

    parser.add_argument('--vocabulary', default='dict.txt', type=str)
    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='dev.json', type=str)

    parser.add_argument(
        '--ckpt_path',
        default='lightning_logs/version_0/checkpoints/epoch=9-step=75069.ckpt',
        type=str
    )
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epochs', default=240, type=int)
    parser.add_argument('--save_top_k', default=10, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--accumulate_grad_batches', default=4, type=int)
    parser.add_argument('--gradient_clip_val', default=5.0, type=float)
    parser.add_argument('--log_every_n_steps', default=100, type=int)

    parser.add_argument('--num_workers', default=0 if platform.system() == 'Windows' else 4, type=int)

    args = parser.parse_args()
    return args


class PyTorchLightningModel(pl.LightningModule):
    def __init__(self,
                 config: str,
                 symbol_table_file: str,
                 train_subset: str = 'train',
                 valid_subset: str = 'valid',
                 batch_size: int = 64,
                 num_workers: int = 0
                 ):
        super().__init__()
        self.train_subset = train_subset
        self.valid_subset = valid_subset
        self.batch_size = batch_size
        self.num_workers = num_workers

        json_str = _jsonnet.evaluate_file(config)
        config = json.loads(json_str)
        print(config)

        self.model = Model.from_json(
            config['model'],
        )

        self.preprocess_list = [
            Preprocess.from_json(json_config) for json_config in config['preprocess']
        ]

        self.collate_fn = CollateFunction(
            symbol_table_file=symbol_table_file,
            preprocess_list=self.preprocess_list
        )

        self.train_dataset = DatasetReader.from_json(
            config['train_dataset_reader']
        ).read(self.train_subset)

        self.train_data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
            prefetch_factor=2,
        )

        self.valid_dataset = DatasetReader.from_json(
            config['valid_dataset_reader']
        ).read(self.valid_subset)

        self.valid_data_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
            prefetch_factor=2,
        )

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor = None,
            text_lengths: torch.Tensor = None,
    ) -> Dict[str, Optional[torch.Tensor]]:

        outputs = self.model.forward(speech, speech_lengths, text, text_lengths)
        return outputs

    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return self.valid_data_loader

    def training_step(self, batch, batch_idx):
        batch_speech, batch_speech_lengths, batch_text, batch_text_lengths = batch
        outputs = self.forward(batch_speech, batch_speech_lengths, batch_text, batch_text_lengths)
        return outputs

    def training_step_end(self, step_output):
        for k, v in step_output.items():
            if k == 'loss':
                continue
            self.log(k, v, prog_bar=True)
        return step_output

    def training_epoch_end(self, outputs):
        return None

    def validation_step(self, batch, batch_idx):
        batch_speech, batch_speech_lengths, batch_text, batch_text_lengths = batch
        outputs = self.forward(batch_speech, batch_speech_lengths, batch_text, batch_text_lengths)
        return outputs

    def validation_step_end(self, step_output):
        for k, v in step_output.items():
            if k == 'loss':
                continue
            self.log('val_{}'.format(k), v, prog_bar=True)
        return step_output

    def validation_epoch_end(self, outputs):
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

        result = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler
            },
        }
        return result


def main():
    args = get_args()
    file_dir = Path(args.file_dir)

    pytorch_lightning_model = PyTorchLightningModel(
        config=args.config,
        symbol_table_file=file_dir / args.vocabulary,
        train_subset=file_dir / args.train_subset,
        valid_subset=file_dir / args.valid_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.ckpt_path is not None:
        pytorch_lightning_model = pytorch_lightning_model.load_from_checkpoint(
            file_dir / args.ckpt_path,
            map_location=torch.device('cpu'),
            **dict(
                config=args.config,
                symbol_table_file=file_dir / args.vocabulary,
                train_subset=file_dir / args.train_subset,
                valid_subset=file_dir / args.valid_subset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )
    pytorch_lightning_model.train()

    index_to_token = {v: k for k, v in pytorch_lightning_model.collate_fn.symbol_table.items()}

    batch_sample = [
        {
            'key': 'test',
            'wav': 'D:/programmer/asr_datasets/aishell/data_aishell/wav/test/S0765/BAC009S0765W0122.wav',
            'txt': ''
        }
    ]
    batch = pytorch_lightning_model.collate_fn(batch_sample)
    batch_speech, batch_speech_lengths, batch_text, batch_text_lengths = batch

    print('ctc greedy search')
    hyps, scores = pytorch_lightning_model.model.ctc_greedy_search(
        speech=batch_speech,
        speech_lengths=batch_speech_lengths,
    )
    for hyp in hyps:
        tokens = list()
        for index in hyp:
            token = index_to_token[index]
            tokens.append(token)

        if tokens[-1] == '<sos/eos>':
            tokens = tokens[:-1]
        text = ''.join(tokens)
        print(text)

    print('encoder decoder search')
    hyps, _ = pytorch_lightning_model.model.recognize(batch_speech, batch_speech_lengths, beam_size=2)
    for hyp in hyps:
        tokens = list()
        for index in hyp.tolist():
            token = index_to_token[index]
            tokens.append(token)

        if tokens[-1] == '<sos/eos>':
            tokens = tokens[:-1]
        text = ''.join(tokens)
        print(text)

    return


if __name__ == '__main__':
    main()
