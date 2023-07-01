#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.wenet.modules.cmvn import GlobalCMVN
from toolbox.wenet.modules.encoders.encoder import Encoder
from toolbox.wenet.modules.subsampling import Subsampling
from toolbox.wenet.nn.activation import Activation
from toolbox.wenet.utils.mask import add_optional_chunk_mask, make_pad_mask


class ConvolutionModule(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        """

        :param channels: int. the number of channels of conv layers.
        :param kernel_size: int. kernel size of conv layers.
        :param activation: torch.nn.Module.
        :param norm: str.
        :param causal: bool. whether use causal convolution or not.
        :param bias: bool.
        """
        # assert check_argument_types()
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.depthwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
            self,
            x: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = f.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = f.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache


class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        """
        FeedForward are applied on each position of the sequence.
        the output dim is same with the input dim.

        :param input_dim: int. input dimension.
        :param hidden_units: int. the number of hidden units.
        :param dropout_rate: float. dropout rate.
        :param activation: torch.nn.Module. activation function.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(input_dim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, input_dim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Forward function.
        :param xs: torch.Tensor. input tensor. shape=(batch_size, max_length, dim).
        :return: output tensor. shape=(batch_size, max_length, dim).
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """
        :param n_head: int. the number of heads.
        :param n_feat: int. the number of features.
        :param dropout_rate: float. dropout rate.
        """
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        transform query, key and value.
        :param query: torch.Tensor. query tensor. shape=(batch_size, time1, n_feat).
        :param key: torch.Tensor. key tensor. shape=(batch_size, time2, n_feat).
        :param value: torch.Tensor. value tensor. shape=(batch_size, time2, n_feat).
        :return:
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self,
                          value: torch.Tensor,
                          scores: torch.Tensor,
                          mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
                          ) -> torch.Tensor:
        """
        compute attention context vector.
        :param value: torch.Tensor. transformed value. shape=(batch_size, n_head, time2, d_k).
        :param scores: torch.Tensor. attention score. shape=(batch_size, n_head, time1, time2).
        :param mask: torch.Tensor. mask. shape=(batch_size, 1, time2) or
                (batch_size, time1, time2), (0, 0, 0) means fake mask.
        :return: torch.Tensor. transformed value. (batch_size, time1, d_model).
                weighted by the attention score (batch_size, time1, time2).
        """
        n_batch = value.size(0)
        # NOTE: When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)

        # NOTE: When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, n_feat)

        return self.linear_out(x)  # (batch, time1, n_feat)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute scaled dot product attention.
        :param query: torch.Tensor. query tensor. shape=(batch_size, time1, n_feat).
        :param key: torch.Tensor. key tensor. shape=(batch_size, time2, n_feat).
        :param value: torch.Tensor. value tensor. shape=(batch_size, time2, n_feat).
        :param mask: torch.Tensor. mask tensor (batch_size, 1, time2) or
                (batch_size, time1, time2), (0, 0, 0) means fake mask.
        :param pos_emb: torch.Tensor. positional embedding tensor. shape=(batch_size, time2, n_feat).
        :param cache: torch.Tensor. cache tensor. shape=(1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == n_feat`
        :return:
        torch.Tensor. output tensor. shape=(batch_size, time1, n_feat).
        torch.Tensor. cache tensor. (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == n_feat`
        """
        q, k, v = self.forward_qkv(query, key, value)

        # NOTE:
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE: We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """
    Paper: https://arxiv.org/abs/1901.02860

    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """
        :param n_head: int. the number of heads.
        :param n_feat: int. the number of features.
        :param dropout_rate: float. dropout rate.
        """
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """
        compute relative positional encoding.
        :param x: torch.Tensor. input tensor. shape=(batch_size, time, size).
        :param zero_triu: bool. if true, return the lower triangular part of the matrix.
        :return: torch.Tensor: output tensor.
        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute 'Scaled Dot Product Attention' with rel. positional encoding.

        :param query: torch.Tensor. query tensor. shape=(batch_size, time1, n_feat).
        :param key: torch.Tensor. key tensor. shape=(batch_size, time2, n_feat).
        :param value: torch.Tensor. value tensor. shape=(batch_size, time2, n_feat).
        :param mask: torch.Tensor. mask tensor (batch_size, 1, time2) or
                (batch_size, time1, time2), (0, 0, 0) means fake mask.
        :param pos_emb: torch.Tensor. positional embedding tensor. shape=(batch_size, time2, n_feat).
        :param cache: torch.Tensor. cache tensor. shape=(1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == n_feat`
        :return:
        torch.Tensor. output tensor. shape=(batch_size, time1, n_feat).
        torch.Tensor. cache tensor. (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == n_feat`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # NOTE:
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE: We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        # (batch, head, time1, time2)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask), new_cache


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 self_attn: torch.nn.Module,
                 feed_forward: torch.nn.Module,
                 dropout_rate: float,
                 normalize_before: bool = True,
                 concat_after: bool = False):
        """
        :param input_dim: int. input dimension.
        :param self_attn: torch.nn.Module. Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
        :param feed_forward: torch.nn.Module. Feed-forward module instance.
                `PositionwiseFeedForward`, instance can be used as the argument.
        :param dropout_rate: float. Dropout rate.
        :param normalize_before: bool.
                True: use layer_norm before each sub-block.
                False: to use layer_norm after each sub-block.
        :param concat_after: bool. Whether to concat attention layer's input and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(input_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(input_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.input_dim = input_dim
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if concat_after:
            self.concat_linear = nn.Linear(input_dim + input_dim, input_dim)
        else:
            self.concat_linear = nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x: torch.Tensor. shape=(batch_size, time, input_dim).
        :param mask: torch.Tensor. mask tensor for the input. shape=(batch_size, time，time),
                (0, 0, 0) means fake mask.
        :param pos_emb: torch.Tensor. just for interface compatibility to ConformerEncoderLayer
        :param mask_pad: torch.Tensor. does not used in transformer layer, just for unified api with conformer.
        :param att_cache: torch.Tensor. cache tensor of the KEY & VALUE
                shape=(batch_size=1, head, cache_t1, d_k * 2), head * d_k == input_dim.
        :param cnn_cache: torch.Tensor. Convolution cache in conformer layer
                (batch_size=1, n_feat, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        :return:
                torch.Tensor: Output tensor (batch_size, time, input_dim).
                torch.Tensor: Mask tensor (batch_size, time, time).
                torch.Tensor: att_cache tensor,
                    (batch_size=1, head, cache_t1 + time, d_k * 2).
                torch.Tensor: cnn_cahce tensor (batch_size=1, input_dim, cache_t2).
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, cache=att_cache)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache


@Encoder.register('transformer_encoder')
class TransformerEncoder(Encoder):
    @staticmethod
    def demo1():
        from toolbox.wenet.modules.subsampling import Conv2dSubsampling4
        from toolbox.wenet.modules.embedding import SinusoidalPositionalEncoding

        batch_size = 3
        seq_len = 1000
        feat_dim = 80
        output_size = 128

        encoder = TransformerEncoder(
            subsampling=Conv2dSubsampling4(
                input_dim=feat_dim,
                output_dim=output_size,
                dropout_rate=0.1,
                positional_encoding=SinusoidalPositionalEncoding(
                    embedding_dim=output_size,
                    dropout_rate=0.1,
                )
            ),
            output_size=128,
            attention_heads=4,
            linear_units=512,
            num_blocks=2,
            use_dynamic_chunk=True,
        )

        xs = torch.randn(size=(batch_size, seq_len, feat_dim))
        xs_lens = torch.randint(int(seq_len * 0.6), seq_len, size=(batch_size,), dtype=torch.long)
        # print(xs.shape)
        # print(xs_lens)

        xs_, masks = encoder.forward(xs, xs_lens)
        # torch.Size([3, 249, 128])
        print(xs_.shape)
        # torch.Size([3, 1, 249])
        print(masks.shape)

        # infer
        xs_, masks = encoder.forward_chunk_by_chunk(
            xs[:1],
            decoding_chunk_size=100,
            num_decoding_left_chunks=1,
        )
        # torch.Size([1, 249, 128])
        print(xs_.shape)
        # torch.Size([1, 1, 249])
        print(masks.shape)
        return

    def __init__(self,
                 subsampling: Subsampling,
                 global_cmvn: GlobalCMVN = None,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 use_dynamic_left_chunk: bool = False,
                 ):
        """
        :param subsampling:
        :param global_cmvn:
        :param output_size: int. dimension of attention.
        :param attention_heads: int. the number of heads of multi head attention.
        :param linear_units: int. the hidden units number of position-wise feed forward.
        :param num_blocks: int. the number of encoder blocks.
        :param dropout_rate: float. dropout rate.
        :param attention_dropout_rate: float. dropout rate in attention.
        :param normalize_before: bool.
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
        :param concat_after: bool. whether to concat attention layer's input and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        :param static_chunk_size: int. chunk size for static chunk training and decoding.
        :param use_dynamic_chunk: bool. whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dynamic chunk size(use_dynamic_chunk = True)
        :param use_dynamic_left_chunk: bool. whether use dynamic left chunk in
                dynamic chunk training.
        """
        super().__init__()
        self.subsampling = subsampling
        self.global_cmvn = global_cmvn

        self._output_size = output_size

        self.normalize_before = normalize_before
        self.concat_after = concat_after

        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=output_size,
                self_attn=MultiHeadedAttention(
                    n_head=attention_heads,
                    n_feat=output_size,
                    dropout_rate=attention_dropout_rate
                ),
                feed_forward=PositionwiseFeedForward(
                    input_dim=output_size,
                    hidden_units=linear_units,
                    dropout_rate=dropout_rate
                ),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after
            ) for _ in range(num_blocks)
        ])

        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        return self._output_size

    def forward(self,
                xs: torch.Tensor,
                xs_lens: torch.Tensor,
                decoding_chunk_size: int = 0,
                num_decoding_left_chunks: int = -1,
                ):
        masks = ~make_pad_mask(xs_lens, max_len=xs.size(1))
        masks = masks.unsqueeze(1)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.subsampling(xs, masks)
        mask_pad = masks  # (batch_size, 1, max_seq_len / subsample_rate)

        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk(self,
                      xs: torch.Tensor,
                      offset: int,
                      required_cache_size: int,
                      att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
                      cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
                      att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward just one chunk.
        :param xs: torch.Tensor. chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + subsample.right_context + 1`
        :param offset: int. current offset in encoder output timestamp.
        :param required_cache_size: int. cache size required for next chunk computation.
                >=0: actual cache size
                <0: means all history cache is required
        :param att_cache: torch.Tensor. cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
        :param cnn_cache: torch.Tensor. cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
        :param att_mask:
        :return:
        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE: Before subsampling, shape(xs) is (b=1, time, mel-dim)
        xs, pos_emb, _ = self.subsampling(xs, tmp_masks, offset)
        # NOTE: After  subsampling, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size

        pos_emb = self.subsampling.position_encoding(
            offset=offset - cache_t1, size=attention_key_size)

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            # NOTE: Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs, att_mask, pos_emb,
                att_cache=att_cache[i:i + 1] if elayers > 0 else att_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache
            )
            # NOTE: After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)

        # NOTE: shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE: shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return xs, r_att_cache, r_cnn_cache

    def forward_chunk_by_chunk(
            self,
            xs: torch.Tensor,
            decoding_chunk_size: int,
            num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk

        subsampling = self.subsampling.subsampling_rate
        context = self.subsampling.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            y, att_cache, cnn_cache = self.forward_chunk(
                chunk_xs, offset, required_cache_size, att_cache, cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
        return ys, masks


class ConformerEncoderLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 self_attn: torch.nn.Module,
                 feed_forward: Optional[nn.Module] = None,
                 feed_forward_macaron: Optional[nn.Module] = None,
                 conv_module: Optional[nn.Module] = None,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 concat_after: bool = False
                 ):
        """
        :param input_dim: int. input dimension.
        :param self_attn: torch.nn.Module. Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
        :param feed_forward: torch.nn.Module. Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
        :param feed_forward_macaron: torch.nn.Module. Additional feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
        :param conv_module: torch.nn.Module. Convolution module instance.
                `ConvolutionModule` instance can be used as the argument.
        :param dropout_rate: float. Dropout rate.
        :param normalize_before: bool.
                True: use layer_norm before each sub-block.
                False: use layer_norm after each sub-block.
        :param concat_after: bool. Whether to concat attention layer's input and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(input_dim, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(input_dim, eps=1e-5)  # for the MHA module

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(input_dim, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(input_dim, eps=1e-5)  # for the CNN module
            self.norm_final = nn.LayerNorm(input_dim, eps=1e-5)  # for the final output of the block

        self.dropout = nn.Dropout(dropout_rate)
        self.input_dim = input_dim
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(input_dim + input_dim, input_dim)
        else:
            self.concat_linear = nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x: torch.Tensor. shape=(batch_size, time, input_dim).
        :param mask: torch.Tensor. mask tensor for the input. shape=(batch_size, time，time),
                (0, 0, 0) means fake mask.
        :param pos_emb: torch.Tensor. positional encoding, must not be None
                for ConformerEncoderLayer.
        :param mask_pad: torch.Tensor. batch padding mask used for conv module.
                shape=(batch_size, 1，time), (0, 0, 0) means fake mask.
        :param att_cache: torch.Tensor. cache tensor of the KEY & VALUE
                (batch_size=1, head, cache_t1, d_k * 2), head * d_k == input_dim.
        :param cnn_cache: torch.Tensor. convolution cache in conformer layer
                shape=(batch_size=1, input_dim, cache_t2)
        :return:
                torch.Tensor: output tensor (batch_size, time, input_dim).
                torch.Tensor: mask tensor (batch_size, time, time).
                torch.Tensor: att_cache tensor. shape=(batch_size=1, head, cache_t1 + time, d_k * 2).
                torch.Tensor: cnn_cache tensor (batch_size, input_dim, cache_t2).
        """
        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb, att_cache)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


@Encoder.register('conformer_encoder')
class ConformerEncoder(TransformerEncoder):
    @staticmethod
    def demo1():
        from toolbox.wenet.modules.subsampling import Conv2dSubsampling4
        from toolbox.wenet.modules.embedding import SinusoidalPositionalEncoding

        batch_size = 3
        seq_len = 1000
        feat_dim = 80
        output_size = 128

        encoder = ConformerEncoder.from_json(
            params={
                'subsampling': {
                    'type': 'subsampling4',
                    'input_dim': feat_dim,
                    'output_dim': output_size,
                    'dropout_rate': 0.1,
                    'positional_encoding': {
                        'type': 'sinusoidal',
                        'embedding_dim': output_size,
                        'dropout_rate': 0.1,
                    },
                },
                'output_size': 128,
                'attention_heads': 4,
                'linear_units': 512,
                'num_blocks': 2,
                'use_dynamic_chunk': True,
            }
        )

        xs = torch.randn(size=(batch_size, seq_len, feat_dim))
        xs_lens = torch.randint(int(seq_len * 0.6), seq_len, size=(batch_size,), dtype=torch.long)
        # print(xs.shape)
        # print(xs_lens)

        xs_, masks = encoder.forward(xs, xs_lens)
        # torch.Size([3, 249, 128])
        print(xs_.shape)
        # torch.Size([3, 1, 249])
        print(masks.shape)

        # infer
        xs_, masks = encoder.forward_chunk_by_chunk(
            xs[:1],
            decoding_chunk_size=100,
            num_decoding_left_chunks=1,
        )
        # torch.Size([1, 249, 128])
        print(xs_.shape)
        # torch.Size([1, 1, 249])
        print(masks.shape)
        return

    def __init__(self,
                 subsampling: Subsampling,
                 global_cmvn: GlobalCMVN = None,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 use_dynamic_left_chunk: bool = False,

                 pos_enc_layer_type: str = "rel_pos",

                 macaron_style: bool = True,
                 activation: str = "swish",
                 use_cnn_module: bool = True,
                 cnn_module_kernel: int = 15,
                 causal: bool = False,
                 cnn_module_norm: str = "batch_norm",
                 ):
        """
        :param subsampling:
        :param global_cmvn:
        :param output_size: int. dimension of attention.
        :param attention_heads: int. the number of heads of multi head attention.
        :param linear_units: int. the hidden units number of position-wise feed forward.
        :param num_blocks: int. the number of encoder blocks.
        :param dropout_rate: float. dropout rate.
        :param attention_dropout_rate: float. dropout rate in attention.
        :param normalize_before: bool.
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
        :param concat_after: bool. whether to concat attention layer's input and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        :param static_chunk_size: int. chunk size for static chunk training and decoding.
        :param use_dynamic_chunk: bool. whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dynamic chunk size(use_dynamic_chunk = True)
        :param use_dynamic_left_chunk: bool. whether use dynamic left chunk in
                dynamic chunk training.
        :param pos_enc_layer_type: str. Encoder positional encoding layer type.
                optional [abs_pos, scaled_abs_pos, rel_pos, no_pos]
        :param macaron_style: bool. Whether to use macaron style for positionwise layer.
        :param activation: str. encoder activation function type.
        :param use_cnn_module: bool. whether to use convolution module.
        :param cnn_module_kernel: int. kernel size of convolution module.
        :param causal: bool. whether to use causal convolution or not.
        :param cnn_module_norm:
        """
        super().__init__(
            subsampling=subsampling,
            global_cmvn=global_cmvn,
            output_size=output_size,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            normalize_before=normalize_before,
            concat_after=concat_after,
            static_chunk_size=static_chunk_size,
            use_dynamic_chunk=use_dynamic_chunk,
            use_dynamic_left_chunk=use_dynamic_left_chunk,
        )
        activation = Activation.by_name(activation)()

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                input_dim=output_size,
                self_attn=MultiHeadedAttention(
                    n_head=attention_heads,
                    n_feat=output_size,
                    dropout_rate=attention_dropout_rate,
                ) if pos_enc_layer_type != 'rel_pos' else RelPositionMultiHeadedAttention(
                    n_head=attention_heads,
                    n_feat=output_size,
                    dropout_rate=attention_dropout_rate,
                ),
                feed_forward=PositionwiseFeedForward(
                    input_dim=output_size,
                    hidden_units=linear_units,
                    dropout_rate=dropout_rate,
                    activation=activation,
                ),
                feed_forward_macaron=PositionwiseFeedForward(
                    input_dim=output_size,
                    hidden_units=linear_units,
                    dropout_rate=dropout_rate,
                    activation=activation,
                ) if macaron_style else None,
                conv_module=ConvolutionModule(
                    channels=output_size,
                    kernel_size=cnn_module_kernel,
                    activation=activation,
                    norm=cnn_module_norm,
                    causal=causal,
                ) if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
            ) for _ in range(num_blocks)
        ])


def demo1():
    # TransformerEncoder.demo1()
    ConformerEncoder.demo1()

    return


if __name__ == '__main__':
    demo1()
