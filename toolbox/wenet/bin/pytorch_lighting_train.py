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

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch.utils.data import Dataset, DataLoader
import _jsonnet

from toolbox.wenet.modules.cmvn import GlobalCMVN
from toolbox.wenet.modules.decoders.transformer_decoder import TransformerDecoder
from toolbox.wenet.modules.encoders.transformer_encoder import ConformerEncoder
from toolbox.wenet.modules.encoders.bilstm_encoder import BiLstmEncoder
from toolbox.wenet.modules.embedding import SinusoidalPositionalEncoding
from toolbox.wenet.models.hybrid_ctc_attention_asr_model import HybridCtcAttentionAsrModel
from toolbox.wenet.modules.loss import CTCLoss, LabelSmoothingLoss
from toolbox.wenet.modules.subsampling import Conv2dSubsampling4
from toolbox.wenet.utils.common import IGNORE_ID
from toolbox.wenet.data.collate_function import CollateFunction
from toolbox.wenet.data.dataset.speech_to_text_json import SpeechToTextJson
from toolbox.wenet.data.dataset.dataset import DatasetReader
from toolbox.wenet.data.preprocess import CjkBpeTokenize, LoadWav, Preprocess, Resample, SpeedPerturb, \
    MapTokensToIds, WaveFormToFbank, SpectrumAug
from toolbox.wenet.data.tokenizers.asr_tokenizer import CjkBpeTokenizer
from toolbox.wenet.utils.file_utils import read_symbol_table
from toolbox.wenet.models.model import Model


class PyTorchLightningModel(pl.LightningModule):
    def __init__(self,
                 train_config_jsonnet_file: str,
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

        # train_config
        json_str = _jsonnet.evaluate_file(train_config_jsonnet_file)
        train_config = json.loads(json_str)

        # model
        self.model = Model.from_json(
            train_config['model'],
        )

        # collate function
        self.preprocess_list = [
            Preprocess.from_json(json_config) for json_config in train_config['preprocess']
        ]
        self.collate_fn = CollateFunction(
            symbol_table_file=symbol_table_file,
            preprocess_list=self.preprocess_list
        )

        # train data
        self.train_dataset = DatasetReader.from_json(
            train_config['train_dataset_reader']
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

        # valid data
        self.valid_dataset = DatasetReader.from_json(
            train_config['valid_dataset_reader']
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
        # accuracy = self._accuracy.get_metric()
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
        # accuracy = self._accuracy.get_metric()
        for k, v in step_output.items():
            if k == 'loss':
                continue
            self.log('val_{}'.format(k), v, prog_bar=True)
        return step_output

    def validation_epoch_end(self, outputs):
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

        result = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler
            },
        }
        return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', default='train_config.jsonnet', type=str)

    # data
    parser.add_argument('--vocabulary', default='dict.txt', type=str)
    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='dev.json', type=str)
    parser.add_argument('--num_workers', default=0 if platform.system() == 'Windows' else 8, type=int)

    # train
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
    return


if __name__ == '__main__':
    main()
