#!/usr/bin/python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset

from toolbox.wenet.common.registrable import Registrable


class DatasetReader(Dataset, Registrable):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return NotImplementedError

    def read(self, filename: str) -> "DatasetReader":
        raise NotImplementedError
