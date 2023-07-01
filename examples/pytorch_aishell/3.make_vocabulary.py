#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import json
from pathlib import Path

from tqdm import tqdm

from toolbox.wenet.data.tokenizers.asr_tokenizer import CjkBpeTokenizer
from toolbox.wenet.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_subset', default='train.json', type=str)
    parser.add_argument('--vocabulary', default='vocabulary', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer = CjkBpeTokenizer()
    counter = Counter()

    with open(args.train_subset, 'r', encoding='utf-8') as fin:
        for row in fin:
            row = json.loads(row)
            txt = row['txt']
            tokens = tokenizer.tokenize(txt)
            counter.update(tokens)

    tokens = list()
    for token, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        tokens.append(token)

    vocabulary = Vocabulary(non_padded_namespaces=['tokens'])
    vocabulary.add_token_to_namespace('<blank>', namespace='tokens')
    vocabulary.add_token_to_namespace('<unk>', namespace='tokens')
    for idx, token in tqdm(enumerate(tokens)):
        vocabulary.add_token_to_namespace(token, namespace='tokens')
    vocabulary.add_token_to_namespace('<sos/eos>', namespace='tokens')

    vocab_size = vocabulary.get_vocab_size(namespace='tokens')
    print('vocab_size: {}'.format(vocab_size))

    vocabulary.save_to_files(args.vocabulary)
    return


if __name__ == '__main__':
    main()
