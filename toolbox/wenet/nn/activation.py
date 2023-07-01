#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn

from toolbox.wenet.common.registrable import Registrable


class Activation(nn.Module, Registrable):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# There are no classes to decorate, so we hack these into Registrable._registry.
# If you want to instantiate it, you can do like this:
# Activation.by_name('relu')()
Registrable._registry[Activation] = {
    "relu": (torch.nn.ReLU, None),
    "relu6": (torch.nn.ReLU6, None),
    "elu": (torch.nn.ELU, None),
    "gelu": (torch.nn.GELU, None),
    "prelu": (torch.nn.PReLU, None),
    "leaky_relu": (torch.nn.LeakyReLU, None),
    "threshold": (torch.nn.Threshold, None),
    "hardtanh": (torch.nn.Hardtanh, None),
    "sigmoid": (torch.nn.Sigmoid, None),
    "tanh": (torch.nn.Tanh, None),
    "log_sigmoid": (torch.nn.LogSigmoid, None),
    "softplus": (torch.nn.Softplus, None),
    "softshrink": (torch.nn.Softshrink, None),
    "softsign": (torch.nn.Softsign, None),
    "tanhshrink": (torch.nn.Tanhshrink, None),
    "selu": (torch.nn.SELU, None),
}


@Activation.register("linear")
class LinearActivation(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@Activation.register("mish")
class MishActivation(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


@Activation.register("swish")
class SwishActivation(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


@Activation.register("gelu_new")
class GeluNew(Activation):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also
    see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
                0.5
                * x
                * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )


@Activation.register("gelu_fast")
class GeluFast(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
