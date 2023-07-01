#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from toolbox.wenet.common.registrable import Registrable


class Decoder(nn.Module, Registrable):

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
        raise NotImplementedError

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
        raise NotImplementedError
