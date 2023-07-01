#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from toolbox.wenet.modules.decoders.decoder import Decoder
from toolbox.wenet.modules.embedding import SinusoidalPositionalEncoding
from toolbox.wenet.modules.encoders.transformer_encoder import MultiHeadedAttention, PositionwiseFeedForward
from toolbox.wenet.utils.mask import make_pad_mask, subsequent_mask


class DecoderLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            self_attn: MultiHeadedAttention,
            src_attn: MultiHeadedAttention,
            feed_forward: PositionwiseFeedForward,
            dropout_rate: float,
            normalize_before: bool = True,
            concat_after: bool = False,
    ):
        """

        :param input_dim: int. input dimension.
        :param self_attn: MultiHeadedAttention. Self-attention module instance.
        :param src_attn: MultiHeadedAttention. Inter-attention module instance.
        :param feed_forward: PositionwiseFeedForward. Feed-forward module instance.
        :param dropout_rate: float. dropout rate.
        :param normalize_before: bool.
                True: use layer_norm before each sub-block.
                False: to use layer_norm after each sub-block.
        :param concat_after: bool. Whether to concat attention layer's input and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        """
        super().__init__()
        self.input_dim = input_dim
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(input_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(input_dim, eps=1e-5)
        self.norm3 = nn.LayerNorm(input_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(input_dim + input_dim, input_dim)
            self.concat_linear2 = nn.Linear(input_dim + input_dim, input_dim)
        else:
            self.concat_linear1 = nn.Identity()
            self.concat_linear2 = nn.Identity()

    def forward(
            self,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        compute decoded features.

        :param tgt: torch.Tensor. input tensor. shape=(batch_size, max_len_out, input_dim).
        :param tgt_mask: torch.Tensor. mask for input tensor. shape=(batch_size, max_len_out).
        :param memory: torch.Tensor. encoded memory. shape=(batch_size, max_len_in, input_dim).
        :param memory_mask: torch.Tensor. encoded memory mask. shape=(batch_size, max_len_in).
        :param cache: torch.Tensor. cached tensors. shape=(batch_size, max_len_out - 1, input_dim).
        :return:
            torch.Tensor: Output tensor (batch_size, max_len_out, input_dim).
            torch.Tensor: Mask for output tensor (batch_size, max_len_out).
            torch.Tensor: Encoded memory (batch_size, max_len_in, input_dim).
            torch.Tensor: Encoded memory mask (batch_size, max_len_in).
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.input_dim,
            ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.input_dim)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0]), dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(
                self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)[0]), dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask)[0])
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


@Decoder.register('transformer_decoder')
class TransformerDecoder(Decoder):
    def __init__(
            self,
            vocab_size: int,
            input_size: int,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            self_attention_dropout_rate: float = 0.0,
            src_attention_dropout_rate: float = 0.0,
            use_output_layer: bool = True,
            normalize_before: bool = True,
            concat_after: bool = False,
    ):
        """
        :param vocab_size: int. output dim.
        :param input_size: int. encoder output size. dimension of attention.
        :param attention_heads: int. the number of heads of multi head attention.
        :param linear_units: int. the hidden units number of position-wise feedforward.
        :param num_blocks: int. the number of decoder blocks.
        :param dropout_rate: float. dropout rate.
        :param positional_dropout_rate: float. dropout rate.
        :param self_attention_dropout_rate: float. dropout rate for attention.
        :param src_attention_dropout_rate:
        :param use_output_layer:
        :param normalize_before:
        :param concat_after:
        """
        super(TransformerDecoder, self).__init__()
        attention_dim = input_size

        self.embedding = torch.nn.Sequential(
            nn.Embedding(vocab_size, input_size),
            SinusoidalPositionalEncoding(input_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks

        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                input_dim=attention_dim,
                self_attn=MultiHeadedAttention(
                    n_head=attention_heads,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate
                ),
                src_attn=MultiHeadedAttention(
                    n_head=attention_heads,
                    n_feat=attention_dim,
                    dropout_rate=src_attention_dropout_rate
                ),
                feed_forward=PositionwiseFeedForward(
                    input_dim=attention_dim,
                    hidden_units=linear_units,
                    dropout_rate=dropout_rate
                ),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
            ) for _ in range(self.num_blocks)
        ])

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
        :param r_ys_in_pad:
        :param reverse_weight:
        :return:
        ys_hat: decoded token score before softmax (batch_size, max_seq_len_out, vocab_size) if use_output_layer is True,
        torch.tensor(0.0): in order to unify api with bidirectional decoder.
        olens: useless. shape=(batch_size, max_seq_len_out).

        """
        tgt = ys_in_pad
        # tgt_mask: (batch_size, 1, max_seq_len_in)
        tgt_mask = ~make_pad_mask(ys_in_lens, tgt.size(1)).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, max_seq_len_in, max_seq_len_in)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (batch_size, max_seq_len_in, max_seq_len_in)
        tgt_mask = tgt_mask & m
        tgt_, _ = self.embedding(tgt)

        # decode
        for layer in self.decoders:
            tgt_, tgt_mask, memory, memory_mask = layer(tgt_, tgt_mask, memory, memory_mask)

        if self.normalize_before:
            tgt_ = self.after_norm(tgt_)
        if self.use_output_layer:
            tgt_ = self.output_layer(tgt_)

        return tgt_, torch.tensor(0.0), tgt_mask.sum(1)

    def forward_one_step(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            cache: Optional[List[torch.Tensor]] = None,
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
        tgt_, _ = self.embedding(tgt)

        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]

            tgt_, tgt_mask, memory, memory_mask = decoder(tgt_,
                                                          tgt_mask,
                                                          memory,
                                                          memory_mask,
                                                          cache=c)
            new_cache.append(tgt_)

        if self.normalize_before:
            tgt_ = self.after_norm(tgt_[:, -1])
        else:
            tgt_ = tgt_[:, -1]

        if self.use_output_layer:
            tgt_ = torch.log_softmax(self.output_layer(tgt_), dim=-1)
        return tgt_, new_cache


@Decoder.register('bitransformer_decoder')
class BiTransformerDecoder(Decoder):
    def __init__(self,
                 vocab_size: int,
                 input_size: int,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 r_num_blocks: int = 0,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 self_attention_dropout_rate: float = 0.0,
                 src_attention_dropout_rate: float = 0.0,
                 use_output_layer: bool = True,
                 normalize_before: bool = True,
                 concat_after: bool = False
                 ):
        super(BiTransformerDecoder, self).__init__()

        self.left_decoder = TransformerDecoder(
            vocab_size, input_size, attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            use_output_layer, normalize_before, concat_after
        )

        self.right_decoder = TransformerDecoder(
            vocab_size, input_size, attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            use_output_layer, normalize_before, concat_after
        )

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
        l_x, _, olens = self.left_decoder(
            memory,
            memory_mask,
            ys_in_pad,
            ys_in_lens
        )

        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(
                memory,
                memory_mask,
                r_ys_in_pad,
                ys_in_lens
            )

        return l_x, r_x, olens

    def forward_one_step(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """

        :param memory:
        :param memory_mask:
        :param tgt:
        :param tgt_mask:
        :param cache:
        :return:
        """
        result = self.left_decoder.forward_one_step(
            memory,
            memory_mask,
            tgt,
            tgt_mask,
            cache
        )
        return result
