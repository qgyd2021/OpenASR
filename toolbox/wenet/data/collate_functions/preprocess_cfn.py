#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import random
import re
from typing import Dict, List, Union

import torch

from toolbox.wenet.data.collate_functions.collate_fn import CollateFunction
from toolbox.wenet.data.preprocess import Preprocess
from toolbox.wenet.utils.common import IGNORE_ID


@CollateFunction.register('preprocess_cfn')
class PreprocessCollateFunction(CollateFunction):
    def __init__(self,
                 ignore_id: int = IGNORE_ID,
                 preprocess_list: List[Preprocess] = None,
                 namespace: str = 'tokens',
                 ):
        super(PreprocessCollateFunction, self).__init__()
        self.ignore_id = ignore_id
        self.preprocess_list = preprocess_list or list()
        self.namespace = namespace

    def feat_pad_or_truncate_ids_by_max_length(self, feat: torch.Tensor, max_length: int):
        seq_len, dim = feat.shape
        if seq_len > max_length:
            result = feat[:max_length]
        else:
            pad_length = max_length - seq_len
            result = torch.cat(tensors=[feat, torch.zeros(size=(pad_length, dim))])
        return result

    def ids_pad_or_truncate_ids_by_max_length(self, ids: torch.Tensor, max_length: int):

        length = ids.size(0)
        if length > max_length:
            result = ids[:max_length]
        else:
            pad_length = max_length - length
            result = torch.cat(tensors=[ids, torch.full(size=(pad_length,), fill_value=self.ignore_id)])
        return result

    def __call__(self, batch_sample: List[dict]):
        batch_ = list()
        for sample in batch_sample:
            for preprocess in self.preprocess_list:
                if self.training is False and preprocess.work_on_eval is False:
                    continue
                sample: Union[dict, None] = preprocess.process(sample)
                if sample is None:
                    continue
            batch_.append(sample)

        feats_max_length = max([sample['feat'].size(0) for sample in batch_])
        ids_max_length = max([sample['index_list'].size(0) for sample in batch_])

        batch_key = list()
        batch_speech = list()
        batch_speech_lengths = list()
        batch_text = list()
        batch_text_lengths = list()

        for sample in batch_:
            key: str = sample['key']
            feat: torch.Tensor = sample['feat']
            ids: torch.Tensor = sample['index_list']

            batch_key.append(key)

            feat_length = feat.size(0)
            ids_length = ids.size(0)

            batch_speech_lengths.append(feat_length)
            batch_text_lengths.append(ids_length)

            feat = self.feat_pad_or_truncate_ids_by_max_length(feat, max_length=feats_max_length)
            ids = self.ids_pad_or_truncate_ids_by_max_length(ids, max_length=ids_max_length)

            batch_speech.append(feat)
            batch_text.append(ids)

        batch_speech = torch.stack(batch_speech, dim=0)
        batch_speech_lengths = torch.tensor(batch_speech_lengths, dtype=torch.long)
        batch_text = torch.stack(batch_text, dim=0)
        batch_text_lengths = torch.tensor(batch_text_lengths, dtype=torch.long)

        # batch_speech = copy.deepcopy(batch_speech)
        # batch_speech_lengths = copy.deepcopy(batch_speech_lengths)
        # batch_text = copy.deepcopy(batch_text)
        # batch_text_lengths = copy.deepcopy(batch_text_lengths)

        return batch_speech, batch_speech_lengths, batch_text, batch_text_lengths
