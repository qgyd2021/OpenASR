#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.wenet.common.registrable import Registrable


class LabelSmoothingLoss(nn.Module, Registrable):
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
        """
        :param predictions: torch.Tensor, shape=(batch_size, seq_len, num_classes).
        :param targets: torch.Tensor, shape=(batch, seq_len). target signal masked with self.padding_id.
        :return: loss: torch.Tensor, The KL loss, scalar float value.
        """
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

        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)

        denom = total if self.normalize_length else batch_size

        result = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        return result


class CTCLoss(nn.Module, Registrable):
    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 dropout_rate: float = 0.0,
                 reduce: bool = True):
        super(CTCLoss, self).__init__()
        reduction_type = "sum" if reduce else "none"

        self.dropout_rate = dropout_rate

        self.projection_layer = nn.Linear(encoder_output_size, vocab_size)
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

    def forward(self,
                hs_pad: torch.Tensor,
                hs_lens: torch.Tensor,
                ys_pad: torch.Tensor,
                ys_lens: torch.Tensor) -> torch.Tensor:
        """
        Calculate CTC loss.
        :param hs_pad: shape=(batch_size, seq_len, hidden_size). batch of padded hidden state sequences.
        :param hs_lens: shape=(batch_size,). batch of lengths of hidden state sequences.
        :param ys_pad: shape=(batch_size, seq_len). batch of padded character id sequence tensor.
        :param ys_lens: shape=(batch_size,). batch of lengths of character sequence.
        :return:
        """
        # hs_pad: (batch_size, seq_len, hidden_size) -> ys_hat: (batch_size, seq_len, vocab_size)
        ys_hat = self.projection_layer(f.dropout(hs_pad, p=self.dropout_rate))

        # ys_hat: (batch_size, seq_len, vocab_size) -> (seq_len, batch_size, vocab_size)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)

        loss = self.ctc_loss(ys_hat, ys_pad, hs_lens, ys_lens)

        # Batch-size average
        loss = loss / ys_hat.size(1)

        return loss

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """
        log_softmax of frame activations
        :param hs_pad: shape=(batch_size, seq_len, vocab_size). batch of padded hidden state sequences.
        :return: shape=(batch_size, seq_len, dim). log softmax applied 3d tensor.
        """
        return f.log_softmax(hs_pad, dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """
        argmax of frame activations
        :param hs_pad: shape=(batch_size, seq_len, vocab_size). batch of padded hidden state sequences.
        :return: shape=(batch_size, seq_len). argmax applied 2d tensor.
        """
        return torch.argmax(hs_pad, dim=2)
