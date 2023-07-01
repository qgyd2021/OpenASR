#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, List

import torch

from toolbox.wenet.common.registrable import Registrable
from toolbox.wenet.data.collate_functions.collate_fn import CollateFunction
from toolbox.wenet.models.model import Model


class Predictor(Registrable):
    def __init__(self,
                 model: Model,
                 collate_fn: CollateFunction,
                 device: torch._C.device = None,
                 ):
        self.model = model.to(device)
        self.collate_fn = collate_fn
        self.device = device or torch.device('cpu')

    def predict_json(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_batch_json(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError
