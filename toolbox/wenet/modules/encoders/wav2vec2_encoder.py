#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from toolbox.wenet.modules.encoders.encoder import Encoder
from toolbox.wenet.utils.mask import add_optional_chunk_mask, make_pad_mask


class Wav2Vec2Encoder(Encoder):
    """
    wav2vec2 use sample rate: 16000
    """
    def __init__(self,
                 wav2vec2_model: Wav2Vec2Model,
                 ):
        super().__init__()
        self.wav2vec2_model = wav2vec2_model

    def forward(self,
                xs: torch.Tensor,
                xs_lens: torch.Tensor,
                decoding_chunk_size: int = 0,
                num_decoding_left_chunks: int = -1,
                ):
        """
        :param xs: shape=(batch_size, seq_len). instead of spectrum, wav2vec2 use raw audio
        :param xs_lens: shape=(batch_size,)
        :param decoding_chunk_size:
        :param num_decoding_left_chunks:
        :return: (xs, masks).
        xs.shape=(batch_size, seq_len / subsample_rate, dim);
        mask.shape=(batch_size, seq_len / subsample_rate);
        """
        if decoding_chunk_size != 0 or num_decoding_left_chunks != -1:
            raise AttributeError('wav2vec2 only supports non-streaming encoding')
        # shape=(batch_size, sequence_length)
        masks = ~make_pad_mask(xs_lens, max_len=xs.size(1))

        # attention_mask: torch.LongTensor, shape=(batch_size, sequence_length).
        encoder_output = self.wav2vec2_model.forward(
            input_values=xs,
            attention_mask=masks
        )
        last_hidden_state = encoder_output.last_hidden_state

        # rescale mask
        scale = last_hidden_state.size(1) / xs.size(1)
        xs_lens_: torch.LongTensor = (xs_lens * scale).clone().detach().long()
        masks_ = ~make_pad_mask(xs_lens_, max_len=last_hidden_state.size(1))
        return last_hidden_state, masks_


@Encoder.register('pretrained_wav2vec2_encoder')
class PretrainedWav2Vec2Encoder(Wav2Vec2Encoder):
    @staticmethod
    def demo1():
        batch_size = 3
        seq_len = 16000 * 3

        pretrained_model_name_or_path = 'D:/programmer/asr_pretrained_model/wav2vec2-large-xlsr-53-chinese-zh-cn'
        encoder = PretrainedWav2Vec2Encoder(
            pretrained_model=pretrained_model_name_or_path,
            requires_grad=True
        )
        print(encoder)

        # instead of spectrum, wav2vec2 use raw audio
        xs = torch.randn(size=(batch_size, seq_len))
        xs_lens = torch.randint(int(seq_len * 0.6), seq_len, size=(batch_size,), dtype=torch.long)
        xs_lens[0] = seq_len
        # print(xs.shape)
        # print(xs_lens)

        xs_, masks = encoder.forward(xs, xs_lens)
        # torch.Size([3, 249, 128])
        print(xs_.shape)
        # torch.Size([3, 1, 249])
        print(masks.shape)
        return

    def __init__(self, pretrained_model: str, requires_grad: bool = False) -> None:
        model = Wav2Vec2Model.from_pretrained(pretrained_model)
        for param in model.parameters():
            param.requires_grad = requires_grad

        super().__init__(wav2vec2_model=model)


def demo1():
    PretrainedWav2Vec2Encoder.demo1()
    return


if __name__ == '__main__':
    demo1()
