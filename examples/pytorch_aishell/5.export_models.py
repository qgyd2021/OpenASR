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
import torch.nn as nn
import _jsonnet

from toolbox.wenet.models import Model
from toolbox.wenet.common.params import Params


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', default='conf/train_conformer.jsonnet', type=str)

    parser.add_argument(
        '--ckpt_path',
        default='file_dir/lightning_logs/version_0/checkpoints/epoch=72-step=137020.ckpt',
        type=str
    )

    parser.add_argument('--state_dict_filename', default='file_dir/pytorch_model.bin', type=str)
    parser.add_argument('--script_model_filename', default='file_dir/script_model.zip', type=str)
    parser.add_argument('--script_quant_model_filename', default='file_dir/script_quant_model.zip', type=str)

    args = parser.parse_args()
    return args


def export_state_dict(model: nn.Module, filename: str = 'pytorch_model.bin'):
    torch.save(
        model.state_dict(),
        filename
    )


def export_jit(
        model: nn.Module,
        script_model_filename: str = 'script_model.zip',
        script_quant_model_filename: str = 'script_quant_model.zip'
):
    script_model = torch.jit.script(obj=model)
    script_model.save(script_model_filename)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    script_quant_model = torch.jit.script(quantized_model)
    script_quant_model.save(script_quant_model_filename)


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
    # print(model)

    # load state dict
    with open(args.ckpt_path, 'rb') as f:
        pytorch_lightning_state_dict = torch.load(f, map_location=torch.device('cpu'))
    state_dict = pytorch_lightning_state_dict['state_dict']
    state_dict = {k[6:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # export state dict
    export_state_dict(model, filename=args.state_dict_filename)
    export_jit(model,
               script_model_filename=args.script_model_filename,
               script_quant_model_filename=args.script_quant_model_filename)

    return


if __name__ == '__main__':
    main()
