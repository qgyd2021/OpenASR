#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.wenet.modules.embedding import PositionalEncoding
from toolbox.wenet.common.registrable import Registrable


class Subsampling(nn.Module, Registrable):
    def __init__(self,
                 positional_encoding: PositionalEncoding,
                 ):
        super().__init__()
        self._positional_encoding = positional_encoding

        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int
                          ) -> torch.Tensor:
        return self._positional_encoding.position_encoding(offset, size)

    def forward(self,
                x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


@Subsampling.register('no_subsampling')
class LinearNoSubsampling(Subsampling):
    """
    Linear transform the input without subsampling
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout_rate: float,
                 positional_encoding: PositionalEncoding
                 ):
        super(LinearNoSubsampling, self).__init__(positional_encoding=positional_encoding)
        self.linear_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.LayerNorm(output_dim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )

        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self,
                x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x: torch.Tensor. Input tensor. shape=(batch_size, time_steps, input_dim).
        :param x_mask: torch.Tensor. shape=(batch_size, 1, time_steps).
        :param offset:
        :return:
        torch.Tensor. linear input tensor. shape=(batch_size, time_steps', output_dim), where time_steps' = time_steps .
        torch.Tensor: positional encoding.
        torch.Tensor. linear input mask. shape=(batch_size, 1, time_steps'), where time_steps' = time_steps .
        """
        x = self.linear_layer(x)
        x, pos_emb = self._positional_encoding.forward(x, offset)
        return x, pos_emb, x_mask


@Subsampling.register('subsampling4')
class Conv2dSubsampling4(Subsampling):
    """
    Convolutional 2D subsampling (to 1/4 length).

    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout_rate: float,
                 positional_encoding: PositionalEncoding
                 ):
        super(Conv2dSubsampling4, self).__init__(positional_encoding=positional_encoding)
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear_layer = torch.nn.Sequential(
            torch.nn.Linear(output_dim * (((input_dim - 1) // 2 - 1) // 2), output_dim)
        )

        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Subsample x.

        :param x: torch.Tensor. Input tensor. shape=(batch_size, time_steps, input_dim).
        :param x_mask: torch.Tensor. Input mask. shape=(batch_size, 1, time_steps).
        :param offset:
        :return:
        torch.Tensor. Subsampled tensor. shape=(batch_size, time_steps', output_dim),
        where time_steps' = time_steps // 4 .
        torch.Tensor: positional encoding.
        torch.Tensor. Subsampled mask. shape=(batch_size, 1, time_steps'),
        where time_steps' = time_steps // 4 .
        """
        # shape=(batch_size, 1, time_steps, input_dim)
        x = x.unsqueeze(1)

        x = self.conv_layer(x)
        b, c, t, f = x.size()

        x = self.linear_layer(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self._positional_encoding.forward(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


@Subsampling.register('subsampling6')
class Conv2dSubsampling6(Subsampling):
    """
    Convolutional 2D subsampling (to 1/6 length).

    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout_rate: float,
                 positional_encoding: PositionalEncoding
                 ):
        super(Conv2dSubsampling6, self).__init__(positional_encoding=positional_encoding)
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, 5, 3),
            torch.nn.ReLU(),
        )
        self.linear_layer = torch.nn.Linear(output_dim * (((input_dim - 1) // 2 - 2) // 3), output_dim)

        self.subsampling_rate = 6
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.right_context = 10

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Subsample x.

        :param x: torch.Tensor. Input tensor. shape=(batch_size, time_steps, input_dim).
        :param x_mask: torch.Tensor. Input mask. shape=(batch_size, 1, time_steps).
        :param offset:
        :return:
        torch.Tensor. Subsampled tensor. shape=(batch_size, time_steps', output_dim),
        where time_steps' = time_steps // 4 .
        torch.Tensor: positional encoding.
        torch.Tensor. Subsampled mask. shape=(batch_size, 1, time_steps'),
        where time_steps' = time_steps // 4 .
        """
        # shape=(batch_size, 1, time_steps, input_dim)
        x = x.unsqueeze(1)

        x = self.conv_layer(x)
        b, c, t, f = x.size()

        x = self.linear_layer(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


@Subsampling.register('subsampling8')
class Conv2dSubsampling8(Subsampling):
    """
    Convolutional 2D subsampling (to 1/8 length).

    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout_rate: float,
                 positional_encoding: PositionalEncoding
                 ):
        super(Conv2dSubsampling8, self).__init__(positional_encoding=positional_encoding)
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear_layer = torch.nn.Linear(output_dim * ((((input_dim - 1) // 2 - 1) // 2 - 1) // 2), output_dim)

        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Subsample x.

        :param x: torch.Tensor. Input tensor. shape=(batch_size, time_steps, input_dim).
        :param x_mask: torch.Tensor. Input mask. shape=(batch_size, 1, time_steps).
        :param offset:
        :return:
        torch.Tensor. Subsampled tensor. shape=(batch_size, time_steps', output_dim),
        where time_steps' = time_steps // 4 .
        torch.Tensor: positional encoding.
        torch.Tensor. Subsampled mask. shape=(batch_size, 1, time_steps'),
        where time_steps' = time_steps // 4 .
        """
        # shape=(batch_size, 1, time_steps, input_dim)
        x = x.unsqueeze(1)

        x = self.conv_layer(x)
        b, c, t, f = x.size()

        x = self.linear_layer(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
