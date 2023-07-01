#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os
from typing import Callable, List

from toolbox.wenet.data.tokenizers.tokenizer import Tokenizer
from toolbox.wenet.data.dataset.dataset import DatasetReader


@DatasetReader.register('speech_to_text_json')
class SpeechToTextJson(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 ):
        self.tokenizer = tokenizer

        self.samples = list()

    def read(self, filename: str) -> DatasetReader:
        samples = list()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                # key, wav, txt
                key = row['key']
                wav = row['wav']
                txt = row.get('txt')
                samples.append({
                    'key': key,
                    'wav': wav,
                    'txt': txt,
                })

        self.samples = samples
        return self

    def __getitem__(self, index):
        instance = self.samples[index]
        return instance

    def __len__(self):
        return len(self.samples)
