#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json

import torch
import torch.nn as nn

from toolbox.wenet.common.registrable import Registrable
from toolbox.wenet.utils.cmvn import load_cmvn


class GlobalCMVN(nn.Module, Registrable):
    def __init__(self,
                 cmvn_file: str,
                 is_json_cmvn: bool = True,
                 norm_var: bool = True
                 ):
        """
        :param cmvn_file: str, global cmvn file.
        :param is_json_cmvn:
        :param norm_var:
        """
        super().__init__()
        self.cmvn_file = cmvn_file

        mean, istd = load_cmvn(cmvn_file, is_json_cmvn)
        mean = torch.from_numpy(mean).float()
        istd = torch.from_numpy(istd).float()

        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        :param x: torch.Tensor. shape=(batch_size, max_length, feat_dim).
        :return: torch.Tensor. normalized feature.
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x
