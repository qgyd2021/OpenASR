#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
import copy
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import math
import os
from pathlib import Path
import platform
import sys
from typing import Dict, List, Optional, Union, Tuple

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./file_dir', type=str)

    parser.add_argument('--src_dict', default='src_dict.txt', type=str)
    parser.add_argument('--tgt_dict', default='tgt_dict.txt', type=str)

    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--valid_subset', default='valid.json', type=str)

    parser.add_argument('--serialization_dir', default='serialization_dir', type=str)
    parser.add_argument('--learning_rate', default=1e-4, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epochs', default=240, type=int)
    parser.add_argument('--save_top_k', default=10, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--gradient_clip_val', default=5.0, type=float)
    parser.add_argument('--log_every_n_steps', default=100, type=int)

    parser.add_argument('--num_workers', default=0 if platform.system() == 'Windows' else 8, type=int)

    args = parser.parse_args()
    return args


def logging_config(file_dir: str):
    format = '[%(asctime)s] %(levelname)s \t [%(filename)s %(lineno)d] %(message)s'
    logging.basicConfig(format=format,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(file_dir, 'log.log'),
        encoding='utf-8',
        when='D',
        interval=1,
        backupCount=7
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(format))
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    return logger


class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional Encoding

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self,
                 embedding_dim: int,
                 dropout_rate: float,
                 max_length: int = 5000,
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        self.x_scale = math.sqrt(self.embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.pe = torch.zeros(self.max_length, self.embedding_dim)
        position = torch.arange(0, self.max_length, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32) *
            - (math.log(10000.0) / self.embedding_dim)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0
                ) -> torch.Tensor:
        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1))
        x = x * self.x_scale + pos_emb
        return self.dropout(x)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          ) -> torch.Tensor:
        if isinstance(offset, int):
            assert offset + size < self.max_length
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size < self.max_length
            pos_emb = self.pe[:, offset:offset + size]
        else:    # for batched streaming decoding on GPU
            # offset. shape=(batch_size,)
            assert torch.max(offset) + size < self.max_length

            # shape=(batch_size, time_steps)
            index = offset.unsqueeze(1) + torch.arange(0, size).to(offset.device)
            flag = index > 0
            # remove negative offset
            index = index * flag
            # shape=(batch_size, time_steps, embedding_dim)
            pos_emb = f.embedding(index, self.pe[0])

        return pos_emb


def add_sos_eos(
        ys_pad: torch.Tensor,
        sos: int, eos: int,
        ignore_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def make_pad_mask(
        lengths: torch.Tensor,
        max_len: int = 0
) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(
        0,
        max_len,
        dtype=torch.int64,
        device=lengths.device
    )
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def pad_list(xs: List[torch.Tensor], pad_value: int):
    n_batch = len(xs)
    max_len = max([x.size(0) for x in xs])
    pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
    pad = pad.fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def pad_sequence(
        sequences: List[torch.Tensor],
        batch_first: bool = False,
        padding_value: float = 0.0
):
    return torch._C._nn.pad_sequence(sequences, batch_first, padding_value)


def reverse_pad_list(ys_pad: torch.Tensor,
                     ys_lens: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
    r_ys_pad = pad_sequence(
        sequences=[(torch.flip(y.int()[:i], [0])) for y, i in zip(ys_pad, ys_lens)],
        batch_first=True,
        padding_value=pad_value
    )
    return r_ys_pad


def th_accuracy(
        pad_outputs: torch.Tensor,
        pad_targets: torch.Tensor,
        ignore_label: int
) -> float:
    pad_outputs = pad_outputs.detach().cpu()
    pad_targets = pad_targets.detach().cpu()

    pad_pred = pad_outputs.view(
        pad_targets.size(0),
        pad_targets.size(1),
        pad_outputs.size(1)
    ).argmax(2)

    mask = pad_targets != ignore_label

    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )

    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


class LabelSmoothingLoss(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.normalize_length = normalize_length

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, _, vocab_size = predictions.shape
        x = predictions.view(-1, self.vocab_size)
        y = targets.view(-1)

        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        ignore = y == self.padding_idx  # (B,)

        total = len(y) - ignore.sum().item()
        target = y.masked_fill(ignore, 0)  # avoid -1 index

        true_dist.scatter_(1, target.unsqueeze(1).long(), self.confidence)

        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)

        denom = total if self.normalize_length else batch_size

        result = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        return result


class Model(nn.Module):
    """
    https://github.com/qqueing/pytorch-G2P/blob/master/model.py
    """
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 encoder_hidden_size: int,
                 encoder_num_layers: int,
                 decoder_hidden_size: int,
                 decoder_num_layers: int,
                 max_seq_length: int = 5000,
                 ignore_id: int = -1,
                 ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layers = decoder_num_layers
        self.max_seq_length = max_seq_length
        self.ignore_id = ignore_id

        self.sos = tgt_vocab_size - 1
        self.eos = tgt_vocab_size - 1

        self.src_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=encoder_hidden_size,
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=decoder_hidden_size,
        )
        self.sinusoidal_positional_encoding = SinusoidalPositionalEncoding(
            embedding_dim=encoder_hidden_size,
            dropout_rate=0.1,
            max_length=max_seq_length
        )
        self.encoder = nn.LSTM(
            input_size=encoder_hidden_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            bias=False,
            batch_first=True,
            bidirectional=True,
        )
        # src projection
        self.src_projection_layer = nn.Linear(in_features=encoder_hidden_size * 2, out_features=encoder_hidden_size)

        self.decoder = nn.LSTMCell(
            input_size=decoder_hidden_size,
            hidden_size=decoder_hidden_size,
        )
        # tgt projection
        self.tgt_projection_layer = nn.Linear(in_features=decoder_hidden_size, out_features=tgt_vocab_size)

        self.att_loss = LabelSmoothingLoss(
            vocab_size=tgt_vocab_size,
            padding_idx=-1,
            smoothing=0.1,
        )

        # decoder
        self.h0 = nn.Parameter(torch.randn(size=(decoder_num_layers, decoder_hidden_size)))
        self.c0 = nn.Parameter(torch.randn(size=(decoder_num_layers, decoder_hidden_size)))

        # memory
        self.key_linear = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        self.value_linear = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        # ht
        self.query_linear = nn.Linear(decoder_hidden_size, decoder_hidden_size, bias=False)

    def encoder_forward(self,
                        src_ids: torch.LongTensor,
                        src_lengths: torch.LongTensor,
                        ) -> torch.Tensor:
        src_embed = self.src_embedding(src_ids)
        src_pos_embed = self.sinusoidal_positional_encoding.forward(src_embed)

        # encoder
        src_packed = pack_padded_sequence(src_pos_embed, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        src_packed, (hn, cn) = self.encoder(src_packed)
        src_encoded, _ = pad_packed_sequence(src_packed, batch_first=True)
        src_encoded = self.src_projection_layer.forward(src_encoded)

        return src_encoded

    def decoder_forward(self,
                        memory: torch.Tensor,
                        memory_mask: torch.Tensor,
                        ys_in_pad: torch.Tensor,
                        ys_in_lens: torch.Tensor,
                        r_ys_in_pad: torch.Tensor = torch.empty(0),
                        reverse_weight: float = 0.0,
                        ):
        tgt = ys_in_pad
        batch_size, max_seq_len_out = ys_in_pad.shape

        # embedding
        tgt_in_embed = self.tgt_embedding(ys_in_pad)
        tgt_in_pos_embed = self.sinusoidal_positional_encoding.forward(tgt_in_embed)

        # attention
        ht_hat = self.forward_attention(memory, memory_mask, self.h0)
        # ct_hat = self.c0.unsqueeze(1)
        # ct_hat = ct_hat.repeat((1, batch_size, 1))

        # ct_hat = self.forward_attention(memory, memory_mask, self.c0)
        # ct_hat = self.c0.unsqueeze(1)
        # ct_hat = ct_hat.repeat((1, batch_size, 1))

        # decode
        tgt_packed = pack_padded_sequence(tgt_in_pos_embed, ys_in_lens.cpu(), batch_first=True, enforce_sorted=False)
        tgt_packed, _ = self.decoder.forward(tgt_packed, hx=ht_hat)
        tgt_decoded, _ = pad_packed_sequence(tgt_packed, batch_first=True)

        # projection
        tgt_decoded = self.tgt_projection_layer.forward(tgt_decoded)

        return tgt_decoded, torch.tensor(0.0)

    def forward(self,
                src_ids: torch.LongTensor,
                src_lengths: torch.LongTensor,
                tgt_ids: torch.LongTensor,
                tgt_lengths: torch.LongTensor,
                ):
        ys_in_pad, ys_out_pad = add_sos_eos(
            tgt_ids, self.sos, self.eos, self.ignore_id)
        ys_in_lens = tgt_lengths + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(tgt_ids, tgt_lengths, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(
            r_ys_pad, self.sos, self.eos, self.ignore_id)

        # encoder
        encoder_out = self.encoder_forward(src_ids, src_lengths)
        encoder_mask = ~make_pad_mask(src_lengths, max_len=src_ids.size(1))
        encoder_mask = encoder_mask.unsqueeze(1)

        # decoder
        decoder_out, _ = self.decoder_forward(
            encoder_out, encoder_mask,
            ys_in_pad, ys_in_lens,
            r_ys_in_pad,
        )

        loss_att = self.att_loss(decoder_out, ys_out_pad)

        # accuracy
        acc_att = th_accuracy(
            pad_outputs=decoder_out.view(-1, self.tgt_vocab_size),
            pad_targets=ys_out_pad,
            ignore_label=self.ignore_id,
        )

        result = {
            'loss': loss_att,
            'loss_att': loss_att.detach(),
            'acc_att': acc_att,
        }
        return result

    def forward_attention(self,
                          memory: torch.Tensor,
                          memory_mask: torch.LongTensor,
                          query: torch.Tensor):
        """
        :param memory: shape=(batch_size, max_seq_len_in, encoder_output_dim)
        :param memory_mask: shape=(batch_size, 1, max_seq_len_in)
        :param query: shape=(batch_size, encoder_output_dim)
        :return:
        """
        # shape=(batch_size, max_seq_len_in, 1, hidden_size)
        value = self.value_linear(memory.unsqueeze(2))

        # shape=(batch_size, max_seq_len_in, hidden_size)
        key = self.key_linear(memory)
        key = key.unsqueeze(2)

        # shape=(batch_size, hidden_size)
        query = self.query_linear(query)
        query = query.view(1, 1, self.decoder_num_layers, self.decoder_hidden_size)

        # shape=(batch_size, max_seq_len_in, num_layers)
        alpha = torch.sum(query * key, dim=-1)
        alpha = alpha.masked_fill(~memory_mask.squeeze(1).unsqueeze(2), -float('inf'))
        alpha = f.softmax(alpha, dim=1)
        alpha = alpha.unsqueeze(-1)

        # shape=(batch_size, num_layers, hidden_size)
        query_hat = torch.sum(alpha * value, dim=1)

        # shape=(num_layers, batch_size, hidden_size)
        query_hat = torch.transpose(query_hat, 0, 1)
        return query_hat


class WhiteSpaceTokenizer(object):
    def tokenize(self, text: str):
        return str(text).strip().split()


class DatasetReader(Dataset):
    def __init__(self):
        self.samples = list()

    def read(self, filename: str):
        samples = list()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)

                src = row['src']
                tgt = row['tgt']

                samples.append({
                    'src': src,
                    'tgt': tgt,
                })

        self.samples = samples
        return self

    def __getitem__(self, index):
        instance = self.samples[index]
        return instance

    def __len__(self):
        return len(self.samples)


class CollateFunction(object):
    def __init__(self,
                 src_symbol_table_file,
                 tgt_symbol_table_file,
                 unk_token: str = '<unk>',
                 pad_token: str = '<blank>',
                 ):
        self.src_symbol_table: Dict[str, int] = read_symbol_table(src_symbol_table_file)
        self.tgt_symbol_table: Dict[str, int] = read_symbol_table(tgt_symbol_table_file)

        self.unk_token = unk_token
        self.pad_token = pad_token

        self.src_tokenizer = WhiteSpaceTokenizer()
        self.tgt_tokenizer = WhiteSpaceTokenizer()

    def pad_or_truncate_ids_by_max_length(self, ids: List[int], max_length: int, pad_idx: int):
        length = len(ids)
        if length > max_length:
            result = ids[:max_length]
        else:
            result = ids + [pad_idx] * (max_length - length)
        return result

    def __call__(self, batch_sample: List[dict]):
        batch_ = list()

        for sample in batch_sample:
            src_text = sample['src']
            tgt_text = sample['tgt']

            src_tokens = self.src_tokenizer.tokenize(src_text)
            tgt_tokens = self.src_tokenizer.tokenize(tgt_text)
            sample['src_tokens'] = src_tokens
            sample['tgt_tokens'] = tgt_tokens

            batch_.append(sample)

        src_max_token_length = max([len(sample['src_tokens']) for sample in batch_])
        tgt_max_token_length = max([len(sample['tgt_tokens']) for sample in batch_])

        batch_src_ids = list()
        batch_src_lens = list()
        batch_tgt_ids = list()
        batch_tgt_lens = list()
        for sample in batch_:
            src_tokens: List[str] = sample['src_tokens']
            tgt_tokens: List[str] = sample['tgt_tokens']

            src_ids: List[int] = [self.src_symbol_table.get(token, self.src_symbol_table[self.unk_token]) for token in src_tokens]
            tgt_ids: List[int] = [self.tgt_symbol_table[token] for token in tgt_tokens]

            src_len = len(src_ids)
            tgt_len = len(tgt_ids)

            src_ids_pad = self.pad_or_truncate_ids_by_max_length(src_ids, max_length=src_max_token_length, pad_idx=self.src_symbol_table[self.pad_token])
            tgt_ids_pad = self.pad_or_truncate_ids_by_max_length(tgt_ids, max_length=tgt_max_token_length, pad_idx=self.tgt_symbol_table[self.pad_token])

            batch_src_ids.append(src_ids_pad)
            batch_src_lens.append(src_len)

            batch_tgt_ids.append(tgt_ids_pad)
            batch_tgt_lens.append(tgt_len)

        batch_src_ids = torch.from_numpy(np.array(batch_src_ids))
        batch_src_lens = torch.from_numpy(np.array(batch_src_lens))
        batch_tgt_ids = torch.from_numpy(np.array(batch_tgt_ids))
        batch_tgt_lens = torch.from_numpy(np.array(batch_tgt_lens))

        return batch_src_ids, batch_src_lens, batch_tgt_ids, batch_tgt_lens


def main():
    args = get_args()
    file_dir = Path(args.file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    serialization_dir = Path(args.serialization_dir)
    serialization_dir.mkdir(parents=True, exist_ok=True)

    logger = logging_config(serialization_dir)

    collate_fn = CollateFunction(
        src_symbol_table_file=file_dir / args.src_dict,
        tgt_symbol_table_file=file_dir / args.tgt_dict,
    )

    train_dataset = DatasetReader().read(file_dir / args.train_subset)

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2,
    )

    valid_dataset = DatasetReader().read(file_dir / args.valid_subset)

    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2,
    )

    model = Model(
        src_vocab_size=30,
        tgt_vocab_size=27,
        encoder_hidden_size=32,
        encoder_num_layers=1,
        decoder_hidden_size=32,
        decoder_num_layers=2,
        max_seq_length=100,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = None
    best_score = None
    patience_count = 0
    score_list = list()

    model_filename_list = list()

    global_step = 0
    for idx_epoch in range(args.max_epochs):
        model.train()

        # train
        total_loss = 0
        total_examples, total_steps = 0, 0
        outputs_dict = defaultdict(float)
        for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch={} (train)'.format(idx_epoch))):
            src_ids, src_lens, tgt_ids, tgt_lens = batch
            src_ids = src_ids.to(device)
            src_lens = src_lens.to(device)
            tgt_ids = tgt_ids.to(device)
            tgt_lens = tgt_lens.to(device)

            outputs = model.forward(
                src_ids,
                src_lens,
                tgt_ids,
                tgt_lens,
            )
            loss = outputs['loss']
            loss.backward()

            total_loss += loss.item()
            total_examples += src_ids.size(0)
            total_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            # lr_scheduler.step()
            global_step += 1

            for k, v in outputs.items():
                outputs_dict[k] = float(v)

            if global_step % args.log_every_n_steps == 0:
                log_str = 'Train Epoch: {}; '.format(idx_epoch)
                for k, v in outputs_dict.items():
                    sub_log_str = '{}: {} '.format(k, v / args.log_every_n_steps)
                    log_str += sub_log_str
                logger.info(log_str)
                outputs_dict = defaultdict(float)
        else:
            log_str = 'Train Epoch: {}; '.format(idx_epoch)
            for k, v in outputs_dict.items():
                sub_log_str = '{}: {} '.format(k, v / args.log_every_n_steps)
                log_str += sub_log_str
            logger.info(log_str)

        # valid
        total_loss = 0
        total_examples, total_steps = 0, 0
        outputs_dict = defaultdict(float)
        for step, batch in enumerate(tqdm(valid_data_loader, desc='Epoch={} (valid)'.format(idx_epoch))):
            src_ids, src_lens, tgt_ids, tgt_lens = batch
            src_ids = src_ids.to(device)
            src_lens = src_lens.to(device)
            tgt_ids = tgt_ids.to(device)
            tgt_lens = tgt_lens.to(device)
            with torch.no_grad():
                outputs = model.forward(
                    src_ids,
                    src_lens,
                    tgt_ids,
                    tgt_lens,
                )
                loss = outputs['loss']
                score: float = outputs['acc_att']
                score_list.append(score)

            total_loss += loss.item()
            total_examples += src_ids.size(0)
            total_steps += 1

            for k, v in outputs.items():
                outputs_dict[k] = float(v)

            if global_step % args.log_every_n_steps == 0:
                log_str = 'Valid Epoch: {}; '.format(idx_epoch)
                for k, v in outputs_dict.items():
                    sub_log_str = '{}: {} '.format(k, v / args.log_every_n_steps)
                    log_str += sub_log_str
                logger.info(log_str)
                outputs_dict = defaultdict(float)
        else:
            log_str = 'Valid Epoch: {}; '.format(idx_epoch)
            for k, v in outputs_dict.items():
                sub_log_str = '{}: {} '.format(k, v / args.log_every_n_steps)
                log_str += sub_log_str
            logger.info(log_str)

        # save model
        model_filename = os.path.join(args.serialization_dir, 'pytorch_model_{}.bin'.format(idx_epoch))
        model_filename_list.append(model_filename)
        if len(model_filename_list) >= args.num_serialized_models_to_keep:
            model_filename_to_delete = model_filename_list.pop(0)
            os.remove(model_filename_to_delete)
        torch.save(model.state_dict(), model_filename)

        # early stop
        score = np.mean(score_list)
        if best_model is None or best_score is None:
            best_model = copy.deepcopy(model)
            best_score = score
            model_filename = os.path.join(args.serialization_dir, 'best.bin')
            torch.save(model.state_dict(), model_filename)
        elif score > best_score:
            best_model = copy.deepcopy(model)
            best_score = score
            model_filename = os.path.join(args.serialization_dir, 'best.bin')
            torch.save(model.state_dict(), model_filename)
            patience_count = 0
        elif patience_count >= args.patience:
            logger.info('Epoch: {}, score did no improve with patience {}. '
                        'early stop.'.format(idx_epoch, args.patience))
            break
        else:
            patience_count += 1

    return


if __name__ == '__main__':
    main()
