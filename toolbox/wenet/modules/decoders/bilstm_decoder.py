#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from toolbox.wenet.modules.decoders.decoder import Decoder
from toolbox.wenet.modules.embedding import SinusoidalPositionalEncoding
from toolbox.wenet.utils.mask import make_pad_mask, subsequent_mask


@Decoder.register('attention_bilstm_decoder')
class AttentionBiLstmDecoder(Decoder):
    """
    https://zhuanlan.zhihu.com/p/88376673?utm_source=ZHShareTargetIDMore
    https://github.com/luopeixiang/im2latex/blob/master/model/model.py
    """
    def __init__(self,
                 vocab_size: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 positional_dropout_rate: float = 0.1,

                 ):
        super(AttentionBiLstmDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            SinusoidalPositionalEncoding(hidden_size, positional_dropout_rate),
        )

        self.h0 = nn.Parameter(torch.randn(size=(num_layers, hidden_size)))
        self.c0 = nn.Parameter(torch.randn(size=(num_layers, hidden_size)))

        # memory
        self.key_linear = nn.Linear(input_size, hidden_size, bias=False)
        self.value_linear = nn.Linear(input_size, hidden_size, bias=False)
        # ht
        self.query_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self._lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self._projection_layer = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward_attention(self,
                          memory: torch.Tensor,
                          memory_mask: torch.LongTensor,
                          ht: torch.Tensor):
        """
        :param memory: shape=(batch_size, max_seq_len_in, encoder_output_dim)
        :param memory_mask: shape=(batch_size, 1, max_seq_len_in)
        :param ht: shape=(batch_size, encoder_output_dim)
        :return:
        """
        # shape=(batch_size, max_seq_len_in, 1, hidden_size)
        value = self.value_linear(memory.unsqueeze(2))

        # shape=(batch_size, max_seq_len_in, hidden_size)
        key = self.key_linear(memory)
        key = key.unsqueeze(2)

        # shape=(batch_size, hidden_size)
        query = self.query_linear(ht)
        query = query.view(1, 1, self.num_layers, self.hidden_size)

        # shape=(batch_size, max_seq_len_in, num_layers)
        alpha = torch.sum(query * key, dim=-1)
        alpha = alpha.masked_fill(~memory_mask.squeeze(1).unsqueeze(2), -float('inf'))
        alpha = f.softmax(alpha, dim=1)
        alpha = alpha.unsqueeze(-1)

        # shape=(batch_size, num_layers, hidden_size)
        ht_hat = torch.sum(alpha * value, dim=1)

        # shape=(num_layers, batch_size, hidden_size)
        ht_hat = torch.transpose(ht_hat, 0, 1)
        return ht_hat

    def forward(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
            r_ys_in_pad: torch.Tensor = torch.empty(0),
            reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param memory: torch.Tensor, shape=(batch_size, max_seq_len_in, encoder_output_dim), encoded memory.
        :param memory_mask: shape=(batch_size, 1, max_seq_len_in), encoder memory mask
        :param ys_in_pad: torch.LongTensor, shape=(batch_size, max_seq_len_out), padded input token ids.
        :param ys_in_lens: shape=(batch_size,), input lengths of this batch.
        :param r_ys_in_pad: torch.LongTensor, shape=(batch_size, max_seq_len_out), used for right to left decoder.
        :param reverse_weight: float. used for right to left decoder
        :return:
        x: decoded token score before softmax.
                shape=(batch_size, max_seq_len_out, vocab_size) if use_output_layer is True,
        r_x: decoded token score before softmax. (right to left decoder).
                shape=(batch_size, max_seq_len_out, vocab_size) if use_output_layer is True,
        olens: useless. shape=(batch_size, max_seq_len_out).

        """
        tgt = ys_in_pad
        batch_size, max_seq_len_out = tgt.shape
        # tgt_mask: (batch_size, 1, max_seq_len_out)
        tgt_mask = ~make_pad_mask(ys_in_lens, max_seq_len_out).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)

        # m: shape=(1, max_seq_len_out, max_seq_len_out)
        m = subsequent_mask(
            tgt_mask.size(-1),
            device=tgt_mask.device
        ).unsqueeze(0)

        # tgt_mask: shape=(batch_size, max_seq_len_out, max_seq_len_out)
        tgt_mask = tgt_mask & m

        tgt_, _ = self.embedding(tgt)

        # shape=(num_layers, batch_size, hidden_size)
        ht_hat = self.forward_attention(memory, memory_mask, self.h0)
        ct_hat = self.c0.unsqueeze(1)
        ct_hat = ct_hat.repeat((1, batch_size, 1))

        # decode
        packed = pack_padded_sequence(tgt_, ys_in_lens, batch_first=True, enforce_sorted=False)
        packed, _ = self._lstm.forward(packed, hx=(ht_hat, ct_hat))
        tgt_, _ = pad_packed_sequence(packed, batch_first=True)

        tgt_ = self._projection_layer.forward(tgt_)

        return tgt_, torch.tensor(0.0), tgt_mask.sum(1)

    def forward_one_step(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward one step. This is only used for decoding.

        :param memory: torch.Tensor, shape=(batch_size, max_seq_len_in, encoder_output_dim), encoded memory.
        :param memory_mask: shape=(batch_size, 1, max_seq_len_in), encoder memory mask.
        :param tgt: shape=(batch_size, max_seq_len_out). input token ids.
        :param tgt_mask: shape=(batch_size, max_seq_len_out). input token mask.
        :param cache: cached output list of (batch, max_time_out-1, size)
        :return:
        y: shape=(batch_size, max_length_out, token).
        cache:
        """
        batch_size, max_seq_len_out = tgt.shape
        # tgt_mask: shape=(batch_size * beam_size, i, i)
        tgt_len = tgt_mask.sum(-1)[:, -1]

        tgt_, _ = self.embedding(tgt)

        if cache is None:
            # shape=(num_layers, batch_size, hidden_size)
            ht_hat = self.forward_attention(memory, memory_mask, self.h0)
            ct_hat = self.c0.unsqueeze(1)
            ct_hat = ct_hat.repeat((1, batch_size, 1))
            hx = (ht_hat, ct_hat)
        else:
            hx = cache

        packed = pack_padded_sequence(tgt_, tgt_len, batch_first=True, enforce_sorted=False)
        packed, new_cache = self._lstm.forward(packed, hx=hx)
        tgt_, _ = pad_packed_sequence(packed, batch_first=True)

        tgt_ = tgt_[:, -1]

        tgt_ = torch.log_softmax(self._projection_layer.forward(tgt_), dim=-1)

        return tgt_, new_cache


def demo1():
    vocab_size = 10
    batch_size = 3
    max_seq_len_in = 12
    encoder_output_dim = 4

    hidden_size = 128
    num_layers = 2

    memory = torch.randn(size=(batch_size, max_seq_len_in, encoder_output_dim))
    memory_lens = torch.tensor(data=[12, 10, 8], dtype=torch.long)
    memory_mask = ~make_pad_mask(lengths=memory_lens, max_len=max_seq_len_in)
    memory_mask = memory_mask.unsqueeze(1)

    ys_in_pad = torch.tensor(data=[[1, 2, 1, 0], [3, 2, 1, 1], [3, 3, 3, 0]], dtype=torch.long)
    ys_in_lens = torch.tensor(data=[3, 4, 3], dtype=torch.long)

    # print(memory_mask)

    decoder = AttentionBiLstmDecoder(
        vocab_size=vocab_size,
        input_size=encoder_output_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    ys_hat, _, _ = decoder.forward(
        memory=memory,
        memory_mask=memory_mask,
        ys_in_pad=ys_in_pad,
        ys_in_lens=ys_in_lens,
    )
    print(ys_hat.shape)
    return


if __name__ == '__main__':
    demo1()
