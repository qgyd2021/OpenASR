#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.wenet.common.registrable import Registrable
from toolbox.wenet.modules.cmvn import GlobalCMVN
from toolbox.wenet.modules.subsampling import Subsampling


class Encoder(nn.Module, Registrable):

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
        raise NotImplementedError

    def forward_chunk_by_chunk(
            self,
            xs: torch.Tensor,
            decoding_chunk_size: int,
            num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward input chunk by chunk with chunk_size like a streaming fashion.


        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.

        :param xs: torch.Tensor. shape=(1, max_len, dim).
        :param decoding_chunk_size: int. decoding chunk size
        :param num_decoding_left_chunks:
        :return:
        """
        raise NotImplementedError
