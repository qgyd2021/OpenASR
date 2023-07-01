#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List

from toolbox.wenet.common.registrable import Registrable


class CollateFunction(Registrable):
    def __init__(self):
        self.training = False

    def __call__(self, batch_sample: List[dict]):
        raise NotImplementedError

    def train(self):
        self.training = True
        return self.training

    def eval(self):
        self.training = False
        return self.training
