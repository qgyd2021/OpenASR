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

    parser.add_argument('--dataset', default='file_dir/test.json', type=str)
    parser.add_argument('--output_file', default='file_dir/eval_test.json', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    json_str = _jsonnet.evaluate_file(args.train_config)
    config: dict = json.loads(json_str)
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        'device': device
    }
    predictor = Predictor.from_params(
        Params(
            params=predictor_params,
            history='predictor'
        ),
        **extra
    )
    # print(predictor)

    with open(args.dataset, 'r', encoding='utf-8') as fin, \
            open(args.output_file, 'w', encoding='utf-8') as fout:
        count = 0
        for row in fin:
            row = json.loads(row)

            outputs = predictor.predict_batch_json(
                [row],
                mode='recognize',
                mode_kwargs={}
            )

            row['hypothesis'] = outputs[0]['hypothesis']
            row = json.dumps(row, ensure_ascii=False)
            fout.write('{}\n'.format(row))

            if count % 1000 == 0:
                print('process count: {}'.format(count))
            count += 1

    return


if __name__ == '__main__':
    main()
