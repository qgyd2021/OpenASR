#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Union

from toolbox.wenet.common.registrable import Registrable


class TrainerBase(Registrable):
    def __init__(self,
                 serialization_dir: str,
                 cuda_device: Union[int, List] = -1) -> None:
        self._serialization_dir = serialization_dir

        if isinstance(cuda_device, list):
            self._multiple_gpu = True
            self._cuda_devices = cuda_device
        else:
            self._multiple_gpu = False
            self._cuda_devices = [cuda_device]

    def train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        raise NotImplementedError
