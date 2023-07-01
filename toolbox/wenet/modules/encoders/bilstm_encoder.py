#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from toolbox.wenet.modules.cmvn import GlobalCMVN
from toolbox.wenet.modules.encoders.encoder import Encoder
from toolbox.wenet.modules.subsampling import Subsampling
from toolbox.wenet.utils.mask import make_pad_mask


@Encoder.register('bilstm_encoder')
class BiLstmEncoder(Encoder):
    @staticmethod
    def demo1():
        from toolbox.wenet.modules.subsampling import Conv2dSubsampling4
        from toolbox.wenet.modules.embedding import SinusoidalPositionalEncoding

        batch_size = 3
        seq_len = 100
        feat_dim = 80
        hidden_size = 128

        encoder = BiLstmEncoder(
            hidden_size=hidden_size,
            num_layers=2,
            output_size=hidden_size,
            subsampling=Conv2dSubsampling4(
                input_dim=feat_dim,
                output_dim=hidden_size,
                dropout_rate=0.1,
                positional_encoding=SinusoidalPositionalEncoding(
                    embedding_dim=hidden_size,
                    dropout_rate=0.1,
                )
            ),
        )

        xs = torch.randn(size=(batch_size, seq_len, feat_dim))
        xs_lens = torch.randint(int(seq_len * 0.6), seq_len, size=(batch_size,), dtype=torch.long)
        # print(xs.shape)
        # print(xs_lens)

        xs, masks = encoder.forward(xs, xs_lens)
        # torch.Size([3, 24, 128])
        print(xs.shape)
        # torch.Size([3, 1, 24])
        print(masks.shape)
        return

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 subsampling: Subsampling,
                 global_cmvn: GlobalCMVN = None,
                 ):
        super(BiLstmEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.subsampling = subsampling
        self.global_cmvn = global_cmvn

        self._lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self._projection_layer = nn.Linear(in_features=hidden_size * 2, out_features=output_size)

    def forward(self,
                xs: torch.Tensor,
                xs_lens: torch.Tensor,
                decoding_chunk_size: int = 0,
                num_decoding_left_chunks: int = -1,
                ):
        """
        :param xs: shape=(batch_size, seq_len, dim)
        :param xs_lens: shape=(batch_size,)
        :param decoding_chunk_size:
        :param num_decoding_left_chunks:
        :return: (xs, masks).
        xs.shape=(batch_size, seq_len / subsample_rate, dim);
        mask.shape=(batch_size, seq_len / subsample_rate);
        """
        masks = ~make_pad_mask(xs_lens, max_len=xs.size(1))
        masks = masks.unsqueeze(1)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.subsampling(xs, masks)
        xs_lens = masks.sum(-1).squeeze(1)

        packed = pack_padded_sequence(xs, xs_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed, (hn, cn) = self._lstm(packed)
        xs, _ = pad_packed_sequence(packed, batch_first=True)

        xs = self._projection_layer(xs)
        return xs, masks


def demo1():
    BiLstmEncoder.demo1()
    return


if __name__ == '__main__':
    demo1()
