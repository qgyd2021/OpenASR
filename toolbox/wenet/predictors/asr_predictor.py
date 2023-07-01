#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, cast, Dict, List

import torch

from toolbox.wenet.common.util import sanitize
from toolbox.wenet.data.collate_functions.collate_fn import CollateFunction
from toolbox.wenet.data.vocabulary import Vocabulary
from toolbox.wenet.models.model import Model
from toolbox.wenet.predictors.predictor import Predictor


@Predictor.register('asr_predictor')
class AsrPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 collate_fn: CollateFunction,
                 vocabulary: Vocabulary,
                 device: torch._C.device,
                 ):
        super().__init__(model=model, collate_fn=collate_fn, device=device)
        self.vocabulary = vocabulary

    def predict_json(self,
                     inputs: Dict[str, Any],
                     mode: str = 'recognize',
                     mode_kwargs: dict = None
                     ) -> Dict[str, Any]:
        outputs = self.predict_batch_json(inputs=[inputs])
        return outputs[0]

    def predict_batch_json(self,
                           inputs: List[Dict[str, Any]],
                           mode: str = 'recognize',
                           mode_kwargs: dict = None
                           ) -> List[Dict[str, Any]]:
        mode_kwargs = mode_kwargs or dict()

        fn = getattr(self.model, mode)

        batch = self.collate_fn(inputs)
        batch = batch[:2]
        batch = [x.to(self.device) for x in batch]

        with torch.no_grad():
            hypotheses, scores = fn(*batch, **mode_kwargs)

        # clear cuda cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        cast(List[List[int]], hypotheses)
        cast(List[float], scores)

        hypotheses = [
            ''.join([self.vocabulary.get_token_from_index(idx, namespace='tokens') for idx in seq])
            for seq in hypotheses
        ]

        outputs = [
            {
                'hypothesis': hypothesis,
                'score': score,
            } for hypothesis, score in zip(hypotheses, scores)
        ]
        return outputs
