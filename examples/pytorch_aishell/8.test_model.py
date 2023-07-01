#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
from typing import Dict, List, Optional

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import torch
import _jsonnet

from toolbox.wenet.models import Model
from toolbox.wenet.common.params import Params
from toolbox.wenet.data.vocabulary import Vocabulary
from toolbox.wenet.data.collate_functions import CollateFunction
from toolbox.wenet.predictors.predictor import Predictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', default='conf/train_conformer.jsonnet', type=str)

    # parser.add_argument('--state_dict_filename', default=None, type=str)
    parser.add_argument('--state_dict_filename', default='file_dir/pytorch_model.bin', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    json_str = _jsonnet.evaluate_file(args.train_config)
    config: dict = json.loads(json_str)
    print(config)

    # model
    model_params = copy.deepcopy(config['model'])
    model = Model.from_params(
        Params(
            params=model_params,
            history='model'
        ),
    )
    if args.state_dict_filename is not None and os.path.exists(args.state_dict_filename):
        print('load state dict: {}'.format(args.state_dict_filename))
        with open(args.state_dict_filename, 'rb') as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    model.eval()
    # print(model)

    # vocabulary
    vocabulary_params = copy.deepcopy(config['vocabulary'])
    vocabulary = Vocabulary.from_params(
        Params(
            params=vocabulary_params,
            history='vocabulary'
        ),
    )
    # print(vocabulary)

    # collate_fn
    collate_fn_params = copy.deepcopy(config['collate_fn'])
    collate_fn = CollateFunction.from_params(
        Params(
            params=collate_fn_params,
            history='collate_fn'
        ),
    )
    collate_fn.train()
    # print(collate_fn)

    # predictor
    predictor_params = copy.deepcopy(config['predictor'])

    extra = {
        'model': model,
        'collate_fn': collate_fn,
        'vocabulary': vocabulary,
    }
    predictor = Predictor.from_params(
        Params(
            params=predictor_params,
            history='predictor'
        ),
        **extra
    )
    print(predictor)

    batch_sample = [
        {
            "key": "BAC009S0724W0126",
            "wav": "D:/programmer/asr_datasets/aishell/data_aishell/wav/dev/S0724/BAC009S0724W0126.wav",
            # "txt": "预计第三季度将陆续有部分股市资金重归楼市",
            "txt": "<unk>"
        },
        {
            "key": "BAC009S0724W0126",
            "wav": "D:/programmer/asr_datasets/aishell/data_aishell/wav/dev/S0724/BAC009S0724W0126.wav",
            # "txt": "预计第三季度将陆续有部分股市资金重归楼市",
            "txt": "<unk>"
        }
    ]
    outputs = predictor.predict_batch_json(
        batch_sample,
        mode='recognize',
        mode_kwargs={}
    )
    print(outputs)

    outputs = predictor.predict_batch_json(
        batch_sample,
        mode='ctc_greedy_search',
        mode_kwargs={}
    )
    print(outputs)

    outputs = predictor.predict_batch_json(
        [batch_sample[0]],
        mode='ctc_prefix_beam_search',
        mode_kwargs={}
    )
    print(outputs)

    outputs = predictor.predict_batch_json(
        [batch_sample[0]],
        mode='attention_rescoring',
        mode_kwargs={}
    )
    print(outputs)
    return


if __name__ == '__main__':
    main()
