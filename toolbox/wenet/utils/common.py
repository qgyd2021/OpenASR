#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import List, Tuple

import torch


IGNORE_ID = -1


def add_sos_eos(
        ys_pad: torch.Tensor,
        sos: int, eos: int,
        ignore_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add <sos> and <eos> labels.

    Examples:
    >>> sos_id = 10
    >>> eos_id = 11
    >>> ignore_id = -1
    >>> ys_pad
    tensor([[ 1,  2,  3,  4,  5],
            [ 4,  5,  6, -1, -1],
            [ 7,  8,  9, -1, -1]], dtype=torch.int32)
    >>> ys_in, ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
    >>> ys_in
    tensor([[10,  1,  2,  3,  4,  5],
            [10,  4,  5,  6, 11, 11],
            [10,  7,  8,  9, 11, 11]])
    >>> ys_out
    tensor([[ 1,  2,  3,  4,  5, 11],
            [ 4,  5,  6, 11, -1, -1],
            [ 7,  8,  9, 11, -1, -1]])

    :param ys_pad: torch.Tensor, shape=(batch_size, seq_len). batch of padded target sequences.
    :param sos: int, index of <sos>.
    :param eos: int, index of <eos>
    :param ignore_id: int, index of padding.
    :return:
    ys_in: torch.Tensor, shape=(batch_size, seq_len + 1).
    ys_out: torch.Tensor, shape=(batch_size, seq_len + 1).

    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def log_add(args: List[int]) -> float:
    """stable log add"""
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """
    Perform padding for the list of tensors.

    Examples:
    >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
    >>> x
    [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
    >>> pad_list(x, 0)
    tensor([[1., 1., 1., 1.],
            [1., 1., 0., 0.],
            [1., 0., 0., 0.]])

    :param xs: List[Tensors], [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
    :param pad_value: float, Value for padding.
    :return:
    padded: shape=(batch_size, max_seq_len, ...), Padded tensor.

    """

    n_batch = len(xs)
    max_len = max([x.size(0) for x in xs])
    pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
    pad = pad.fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def pad_sequence(
        sequences: List[torch.Tensor],
        batch_first: bool = False,
        padding_value: float = 0.0
):
    """
    Pad a list of variable length Tensors with `padding_value`.


    `pad_sequence` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size `L x *` and if batch_first is False, and `T x B x *`
    otherwise.

    `B` is batch size. It is equal to the number of elements in `sequences`.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
    >>> from torch.nn.utils.rnn import pad_sequence
    >>> a = torch.ones(25, 300)
    >>> b = torch.ones(22, 300)
    >>> c = torch.ones(15, 300)
    >>> pad_sequence([a, b, c]).size()
    torch.Size([25, 3, 300])

    Note:
    This function returns a Tensor of size `T x B x *` or `B x T x *`
    where `T` is the length of the longest sequence. This function assumes
    trailing dimensions and type of all the Tensors in sequences are same.

    :param sequences: List[Tensor], list of variable length sequences.
    :param batch_first: bool, optional, output will be in `B x T x *` if True, or in `T x B x *` otherwise
    :param padding_value: float, optional, value for padded elements. Default: 0.
    :return: Tensor of size `T x B x *` if :attr:`batch_first` is `False`. Tensor of size ``B x T x *`` otherwise

    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return torch._C._nn.pad_sequence(sequences, batch_first, padding_value)


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def reverse_pad_list(ys_pad: torch.Tensor,
                     ys_lens: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
    """
    Reverse padding for the list of tensors.

    Examples:
    >>> x
    tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
    >>> pad_list(x, 0)
    tensor([[4, 3, 2, 1],
            [7, 6, 5, 0],
            [9, 8, 0, 0]])

    :param ys_pad: tensor, shape=(batch_size, max_token_len), The padded tensor
    :param ys_lens: tensor, shape=(batch_size,), The lens of token seqs
    :param pad_value: int, Value for padding.
    :return: shape=(batch_size, max_token_len). padded tensor.
    """

    r_ys_pad = pad_sequence(
        sequences=[(torch.flip(y.int()[:i], [0])) for y, i in zip(ys_pad, ys_lens)],
        batch_first=True,
        padding_value=pad_value
    )
    return r_ys_pad


def th_accuracy(
        pad_outputs: torch.Tensor,
        pad_targets: torch.Tensor,
        ignore_label: int
) -> float:
    """
    Calculate accuracy.

    :param pad_outputs: torch.Tensor, shape=(batch_size * max_seq_len, dim), prediction tensors.
    :param pad_targets: torch.LongTensor, shape=(batch_size, max_seq_len), target label tensors.
    :param ignore_label: int, ignore label id.
    :return: float, between (0.0 - 1.0), accuracy value.
    """
    pad_outputs = pad_outputs.detach().cpu()
    pad_targets = pad_targets.detach().cpu()

    pad_pred = pad_outputs.view(
        pad_targets.size(0),
        pad_targets.size(1),
        pad_outputs.size(1)
    ).argmax(2)

    mask = pad_targets != ignore_label

    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )

    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)

