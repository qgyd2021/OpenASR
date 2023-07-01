#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.wenet.common.registrable import Registrable


class PositionalEncoding(nn.Module, Registrable):
    def __init__(self,
                 embedding_dim: int,
                 dropout_rate: float,
                 max_length: int = 5000,
                 reverse: bool = False
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.reverse = reverse

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int
                          ) -> torch.Tensor:
        raise NotImplementedError


@PositionalEncoding.register('sinusoidal')
class SinusoidalPositionalEncoding(PositionalEncoding):
    """
    Positional Encoding

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    @staticmethod
    def demo1():
        batch_size = 2
        time_steps = 10
        embedding_dim = 64

        pe = SinusoidalPositionalEncoding(
            embedding_dim=embedding_dim,
            dropout_rate=0.1,
        )

        x = torch.randn(size=(batch_size, time_steps, embedding_dim))

        x, pos_emb = pe.forward(x)

        # torch.Size([2, 10, 64])
        print(x.shape)
        # torch.Size([1, 10, 64])
        print(pos_emb.shape)
        return

    @staticmethod
    def demo2():
        batch_size = 2
        time_steps = 10
        embedding_dim = 64

        pe = SinusoidalPositionalEncoding(
            embedding_dim=embedding_dim,
            dropout_rate=0.1,
        )

        x = torch.randn(size=(batch_size, time_steps, embedding_dim))
        offset = torch.randint(low=3, high=7, size=(batch_size,))
        x, pos_emb = pe.forward(x, offset=offset)

        # tensor([3, 4])
        print(offset)
        # torch.Size([2, 10, 64])
        print(x.shape)
        # torch.Size([2, 10, 64])
        print(pos_emb.shape)
        return

    def __init__(self,
                 embedding_dim: int,
                 dropout_rate: float,
                 max_length: int = 5000,
                 reverse: bool = False
                 ):
        super().__init__(embedding_dim, dropout_rate, max_length, reverse=reverse)
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
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add positional encoding.
        :param x: torch.Tensor. Input. shape=(batch_size, time_steps, ...).
        :param offset: int or torch.Tensor. position offset.
        :return:
        torch.Tensor. Encoded tensor. shape=(batch_size, time_steps, ...).
        torch.Tensor. for compatibility to RelPositionalEncoding. shape=(1, time_steps, ...).
        """
        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.x_scale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True
                          ) -> torch.Tensor:
        """
        For getting encoding in a streaming fashion.

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        :param offset: int or torch.Tensor. start offset.
        :param size: int. required size of position encoding.
        :param apply_dropout:
        :return: torch.Tensor. Corresponding encoding.
        """
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

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


@PositionalEncoding.register('relative')
class RelPositionalEncoding(SinusoidalPositionalEncoding):
    """
    Relative positional encoding module.

    See : Appendix B in https://arxiv.org/abs/1901.02860


    """
    def __init__(self,
                 embedding_dim: int,
                 dropout_rate: float,
                 max_length: int = 5000,
                 reverse: bool = False
                 ):
        super().__init__(embedding_dim, dropout_rate, max_length, reverse=True)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute positional encoding.
        :param x: torch.Tensor. Input. shape=(batch_size, time_steps, ...).
        :param offset:
        :return:
        torch.Tensor. Encoded tensor. shape=(batch_size, time_steps, ...).
        torch.Tensor. Positional embedding tensor. shape=(1, time_steps, ...).
        """
        self.pe = self.pe.to(x.device)
        x = x * self.x_scale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


@PositionalEncoding.register('no_positional_encoding')
class NoPositionalEncoding(PositionalEncoding):
    """
    No position encoding
    """
    def __init__(self,
                 embedding_dim: int,
                 dropout_rate: float,
                 max_length: int = 5000,
                 reverse: bool = False
                 ):
        super().__init__(embedding_dim, dropout_rate, max_length, reverse=reverse)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Just return zero vector for interface compatibility
        """
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        return self.dropout(x), pos_emb

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int
                          ) -> torch.Tensor:
        return torch.zeros(1, size, self.d_model)
