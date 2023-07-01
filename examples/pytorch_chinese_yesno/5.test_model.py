#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import platform
from typing import Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch.utils.data import Dataset, DataLoader

from toolbox.wenet.models.cmvn import GlobalCMVN
from toolbox.wenet.models.decoders.bilstm_decoder import AttentionBiLstmDecoder
from toolbox.wenet.models.encoders.bilstm_encoder import BiLstmEncoder
from toolbox.wenet.models.embedding import SinusoidalPositionalEncoding
from toolbox.wenet.models.hybrid_ctc_attention_asr_model import HybridCtcAttentionAsrModel
from toolbox.wenet.models.loss import CTCLoss, LabelSmoothingLoss
from toolbox.wenet.models.subsampling import Conv2dSubsampling4
from toolbox.wenet.utils.common import IGNORE_ID
from toolbox.wenet.utils.data.dataset.speech_to_text_json import SpeechToTextJson
from toolbox.wenet.utils.data.preprocess import CjkBpeTokenize, LoadWav, Resample, Preprocess, MapTokensToIds, WaveFormToFbank
from toolbox.wenet.utils.data.tokenizers.asr_tokenizer import CjkBpeTokenizer
from toolbox.wenet.utils.file_utils import read_symbol_table


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./file_dir', type=str)

    parser.add_argument('--vocabulary', default='dict.txt', type=str)
    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='valid.json', type=str)
    parser.add_argument('--global_cmvn', default='global_cmvn', type=str)

    parser.add_argument(
        '--ckpt_path',
        # default=None,
        default='lightning_logs/version_0/checkpoints/epoch=999-step=8000.ckpt',
        type=str
    )
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--save_top_k', default=10, type=int)
    parser.add_argument('--patience', default=5, type=int)

    args = parser.parse_args()
    return args


class CollateFunction(object):
    def __init__(self,
                 symbol_table: Dict[str, int],
                 unk_token: str = '<unk>',
                 pad_token: str = '<blank>',
                 preprocess_list: List[Preprocess] = None,
                 ):
        self.symbol_table = symbol_table
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.preprocess_list = preprocess_list or list()

    def feat_pad_or_truncate_ids_by_max_length(self, feat: torch.Tensor, max_length: int):
        seq_len, dim = feat.shape
        if seq_len > max_length:
            result = feat[:max_length]
        else:
            pad_length = max_length - seq_len
            result = torch.cat(tensors=[feat, torch.zeros(size=(pad_length, dim))])
        return result

    def ids_pad_or_truncate_ids_by_max_length(self, ids: torch.Tensor, max_length: int):
        pad_idx = self.symbol_table[self.pad_token]

        length = ids.size(0)
        if length > max_length:
            result = ids[:max_length]
        else:
            pad_length = max_length - length
            result = torch.cat(tensors=[ids, torch.full(size=(pad_length,), fill_value=pad_idx)])
        return result

    def __call__(self, batch_sample: List[dict]):
        batch_ = list()
        for sample in batch_sample:
            for preprocess in self.preprocess_list:
                sample = preprocess.process(sample)
            batch_.append(sample)

        feats_max_length = max([sample['feat'].size(0) for sample in batch_])
        ids_max_length = max([sample['index_list'].size(0) for sample in batch_])

        batch_key = list()
        batch_speech = list()
        batch_speech_lengths = list()
        batch_text = list()
        batch_text_lengths = list()

        for sample in batch_:
            key: str = sample['key']
            feat: torch.Tensor = sample['feat']
            ids: torch.Tensor = sample['index_list']

            batch_key.append(key)

            feat_length = feat.size(0)
            ids_length = ids.size(0)

            batch_speech_lengths.append(feat_length)
            batch_text_lengths.append(ids_length)

            feat = self.feat_pad_or_truncate_ids_by_max_length(feat, max_length=feats_max_length)
            ids = self.ids_pad_or_truncate_ids_by_max_length(ids, max_length=ids_max_length)

            batch_speech.append(feat)
            batch_text.append(ids)

        batch_speech = torch.stack(batch_speech, dim=0)
        batch_speech_lengths = torch.tensor(batch_speech_lengths, dtype=torch.long)
        batch_text = torch.stack(batch_text, dim=0)
        batch_text_lengths = torch.tensor(batch_text_lengths, dtype=torch.long)

        return batch_speech, batch_speech_lengths, batch_text, batch_text_lengths


class Model(pl.LightningModule):
    def __init__(self,
                 symbol_table_file: str,
                 train_subset: str = 'train',
                 valid_subset: str = 'valid',
                 cmvn_file: str = 'global_cmvn',
                 batch_size: int = 64,
                 feat_dim: int = 80,
                 encoder_hidden_size: int = 128,
                 encoder_output_size: int = 128,
                 encoder_num_layers: int = 2,
                 decoder_hidden_size: int = 128,
                 decoder_num_layers: int = 2,
                 ):
        super().__init__()
        self.train_subset = train_subset
        self.valid_subset = valid_subset
        self.batch_size = batch_size

        self.symbol_table: Dict[str, int] = read_symbol_table(symbol_table_file)
        vocab_size = len(self.symbol_table)

        self.asr_model = HybridCtcAttentionAsrModel(
            vocab_size=vocab_size,
            encoder=BiLstmEncoder(
                hidden_size=encoder_hidden_size,
                num_layers=encoder_num_layers,
                output_size=encoder_output_size,
                subsampling=Conv2dSubsampling4(
                    input_dim=feat_dim,
                    output_dim=encoder_hidden_size,
                    dropout_rate=0.1,
                    positional_encoding=SinusoidalPositionalEncoding(
                        embedding_dim=encoder_hidden_size,
                        dropout_rate=0.1,
                    )
                ),
                global_cmvn=GlobalCMVN(
                    cmvn_file=cmvn_file,
                    is_json_cmvn=True
                ),
            ),
            decoder=AttentionBiLstmDecoder(
                vocab_size=vocab_size,
                input_size=encoder_output_size,
                hidden_size=decoder_hidden_size,
                num_layers=decoder_num_layers,
            ),
            ctc_loss=CTCLoss(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
            ),
            att_loss=LabelSmoothingLoss(
                vocab_size=vocab_size,
                padding_idx=IGNORE_ID,
                smoothing=0.1,
            ),
        )
        self.collate_fn = CollateFunction(
            symbol_table=self.symbol_table,
            preprocess_list=[
                LoadWav(),
                Resample(resample_rate=8000),
                CjkBpeTokenize(),
                MapTokensToIds(
                    symbol_table=self.symbol_table,
                ),
                WaveFormToFbank(num_mel_bins=23)
            ]
        )

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor = None,
            text_lengths: torch.Tensor = None,
    ) -> Dict[str, Optional[torch.Tensor]]:

        outputs = self.asr_model.forward(speech, speech_lengths, text, text_lengths)
        return outputs


def main():
    args = get_args()
    file_dir = Path(args.file_dir)

    model = Model(
        symbol_table_file=file_dir / args.vocabulary,
        train_subset=file_dir / args.train_subset,
        valid_subset=file_dir / args.valid_subset,
        cmvn_file=file_dir / args.global_cmvn,
        batch_size=args.batch_size,
        feat_dim=23,
        encoder_hidden_size=32,
        encoder_output_size=16,
        encoder_num_layers=2,
        decoder_hidden_size=32,
        decoder_num_layers=2,
    )

    model: Model = model.load_from_checkpoint(
        file_dir / args.ckpt_path,
        map_location=torch.device('cpu'),
        **{
            'symbol_table_file': file_dir / args.vocabulary,
            'train_subset': file_dir / args.train_subset,
            'valid_subset': file_dir / args.valid_subset,
            'cmvn_file': file_dir / args.global_cmvn,
            'batch_size': args.batch_size,
            'feat_dim': 23,
            'encoder_hidden_size': 32,
            'encoder_output_size': 16,
            'encoder_num_layers': 2,
            'decoder_hidden_size': 32,
            'decoder_num_layers': 2,
        }
    )
    model.eval()

    index_to_token = {v: k for k, v in model.symbol_table.items()}

    samples = [
        {
            "key": "s1_t14",
            "wav": "file_dir/yesno_cn/s1_t14.wav",
            "txt": "可以"
        },
        {
            "key": "s1_t15",
            "wav": "file_dir/yesno_cn/s1_t15.wav",
            "txt": "可以"
        },
        {
            "key": "s1_t29",
            "wav": "file_dir/yesno_cn/s1_t29.wav",
            "txt": "不行"
        },
        {
            "key": "s1_t30",
            "wav": "file_dir/yesno_cn/s1_t30.wav",
            "txt": "不行"
        }
    ]

    batch = model.collate_fn(samples)
    batch_speech, batch_speech_lengths, batch_text, batch_text_lengths = batch

    print('ctc greedy search')
    hyps, _ = model.asr_model.ctc_greedy_search(batch_speech, batch_speech_lengths)
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
    hyps, _ = model.asr_model.recognize(batch_speech, batch_speech_lengths, beam_size=2)
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


