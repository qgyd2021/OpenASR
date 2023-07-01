#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as f

from toolbox.wenet.common.registrable import Registrable


class Model(nn.Module, Registrable):
    def __init__(self):
        super().__init__()
