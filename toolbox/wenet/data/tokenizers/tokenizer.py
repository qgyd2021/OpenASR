#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List

from toolbox.wenet.common.registrable import Registrable


class Tokenizer(Registrable):
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError


if __name__ == '__main__':
    pass
