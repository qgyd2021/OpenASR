#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import platform
import sys
from typing import Dict, List, Optional, Set
import unicodedata

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_words_per_line', default=sys.maxsize, type=int)
    parser.add_argument('--remove_tag', default=True, type=bool)
    parser.add_argument('--do_lowercase', default=True, type=bool)
    parser.add_argument('--cluster_file', default=None, type=str)
    parser.add_argument('--split_file', default=None, type=str, help='')
    parser.add_argument('--ignore_file', default=None, type=str)
    parser.add_argument('--to_char', default=True, type=bool)
    parser.add_argument('--verbose', default=2, type=int,
                        help='log verbose level, bigger and more detailed. the max is 2.')
    parser.add_argument('--padding_symbol', default='space', type=str)

    parser.add_argument('--eval_file', default='file_dir/eval_test.json', type=str)

    args = parser.parse_args()
    return args


space_list = [' ', '\t', '\r', '\n']

punctuations = ['!', ',', '?', '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰',  '『', '』', '《', '》']


def characterize(string: str) -> List[str]:
    result = list()
    i = 0
    while i < len(string):
        char = string[i]
        if char in punctuations:
            i += 1
            continue
        cat1 = unicodedata.category(char)

        # https://unicodebook.readthedocs.io/unicode.html#unicode-categories
        if cat1 == 'Zs' or cat1 == 'Cn' or char in space_list:   # space or not assigned
            i += 1
            continue
        if cat1 == 'Lo':   # letter-other
            result.append(char)
            i += 1
        else:
            # some input looks like: <unk><noise>, we want to separate it to two words.
            sep = ' '
            if char == '<':
                sep = '>'
            j = i + 1
            while j < len(string):
                c = string[j]
                if ord(c) >= 128 or (c in space_list) or (c == sep):
                    break
                j += 1
            if j < len(string) and string[j] == '>':
                j += 1
            result.append(string[i:j])
            i = j
    return result


def stripoff_tags(x: str):
    if len(x) == 0:
        return ''

    chars = list()
    i = 0
    length = len(x)
    while i < length:
        if x[i] == '<':
            while i < length and x[i] != '>':
                i += 1
            i += 1
        else:
            chars.append(x[i])
            i += 1
    result = ''.join(chars)
    return result


def normalize(sentence: List[str],
              ignore_words: Set[str],
              do_lowercase: bool,
              split: Dict[str, List[str]] = None,
              remove_tag: bool = True) -> List[str]:
    new_sentence = []
    for token in sentence:
        x = token
        if not do_lowercase:
            x = x.upper()
        if x in ignore_words:
            continue
        if remove_tag:
            x = stripoff_tags(x)
        if not x:
            continue
        if split and x in split:
            new_sentence += split[x]
        else:
            new_sentence.append(x)
    return new_sentence


class Calculator:
    def __init__(self):
        self.data = dict()
        self.space = list()
        self.cost = {
            'cor': 0,
            'sub': 1,
            'del': 1,
            'ins': 1,
        }

    def calculate(self, lab: List[str], rec: List[str]):
        # Initialization
        lab.insert(0, '')
        rec.insert(0, '')
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            for element in row:
                element['dist'] = 0
                element['error'] = 'non'
            while len(row) < len(rec):
                row.append({'dist': 0, 'error': 'non'})
        for i in range(len(lab)):
            self.space[i][0]['dist'] = i
            self.space[i][0]['error'] = 'del'
        for j in range(len(rec)):
            self.space[0][j]['dist'] = j
            self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        for token in lab:
            if token not in self.data and len(token) > 0:
                self.data[token] = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in rec:
            if token not in self.data and len(token) > 0:
                self.data[token] = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}

        # Computing edit distance
        for i, lab_token in enumerate(lab):
            for j, rec_token in enumerate(rec):
                if i == 0 or j == 0:
                    continue
                min_dist = sys.maxsize
                min_error = 'none'
                dist = self.space[i-1][j]['dist'] + self.cost['del']
                error = 'del'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j-1]['dist'] + self.cost['ins']
                error = 'ins'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token:
                    dist = self.space[i-1][j-1]['dist'] + self.cost['cor']
                    error = 'cor'
                else:
                    dist = self.space[i-1][j-1]['dist'] + self.cost['sub']
                    error = 'sub'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                self.space[i][j]['dist'] = min_dist
                self.space[i][j]['error'] = min_error

        # Tracing back
        result = {'lab': [], 'rec': [], 'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            if self.space[i][j]['error'] == 'cor':    # correct
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
                    result['all'] = result['all'] + 1
                    result['cor'] = result['cor'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'sub':    # substitution
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
                    result['all'] = result['all'] + 1
                    result['sub'] = result['sub'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'del':    # deletion
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
                    result['all'] = result['all'] + 1
                    result['del'] = result['del'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, "")
                i = i - 1
            elif self.space[i][j]['error'] == 'ins':    # insertion
                if len(rec[j]) > 0:
                    self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
                    result['ins'] = result['ins'] + 1
                result['lab'].insert(0, "")
                result['rec'].insert(0, rec[j])
                j = j - 1
            elif self.space[i][j]['error'] == 'non':    # starting point
                break
            else:    # shouldn't reach here
                print('this should not happen , i = {i} , j = {j} , error = {error}'.format(
                    i=i, j=j, error=self.space[i][j]['error']))
        return result

    def overall(self):
        result = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in self.data:
            result['all'] = result['all'] + self.data[token]['all']
            result['cor'] = result['cor'] + self.data[token]['cor']
            result['sub'] = result['sub'] + self.data[token]['sub']
            result['ins'] = result['ins'] + self.data[token]['ins']
            result['del'] = result['del'] + self.data[token]['del']
        return result

    def cluster(self, data):
        result = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in data:
            if token in self.data:
                result['all'] = result['all'] + self.data[token]['all']
                result['cor'] = result['cor'] + self.data[token]['cor']
                result['sub'] = result['sub'] + self.data[token]['sub']
                result['ins'] = result['ins'] + self.data[token]['ins']
                result['del'] = result['del'] + self.data[token]['del']
        return result

    def keys(self):
        return list(self.data.keys())


def width(string):
    return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)


def default_cluster(word: str):
    unicode_names = [unicodedata.name(char) for char in word]
    for i in reversed(range(len(unicode_names))):
        if unicode_names[i].startswith('DIGIT'):  # 1
            unicode_names[i] = 'Number'  # 'DIGIT'
        elif (unicode_names[i].startswith('CJK UNIFIED IDEOGRAPH') or
              unicode_names[i].startswith('CJK COMPATIBILITY IDEOGRAPH')):
            # 明 / 郎
            unicode_names[i] = 'Mandarin'  # 'CJK IDEOGRAPH'
        elif (unicode_names[i].startswith('LATIN CAPITAL LETTER') or
              unicode_names[i].startswith('LATIN SMALL LETTER')):
            # A / a
            unicode_names[i] = 'English'  # 'LATIN LETTER'
        elif unicode_names[i].startswith('HIRAGANA LETTER'):  # は こ め
            unicode_names[i] = 'Japanese'  # 'GANA LETTER'
        elif (unicode_names[i].startswith('AMPERSAND') or
              unicode_names[i].startswith('APOSTROPHE') or
              unicode_names[i].startswith('COMMERCIAL AT') or
              unicode_names[i].startswith('DEGREE CELSIUS') or
              unicode_names[i].startswith('EQUALS SIGN') or
              unicode_names[i].startswith('FULL STOP') or
              unicode_names[i].startswith('HYPHEN-MINUS') or
              unicode_names[i].startswith('LOW LINE') or
              unicode_names[i].startswith('NUMBER SIGN') or
              unicode_names[i].startswith('PLUS SIGN') or
              unicode_names[i].startswith('SEMICOLON')):
            # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
            del unicode_names[i]
        else:
            return 'Other'
    if len(unicode_names) == 0:
        return 'Other'
    if len(unicode_names) == 1:
        return unicode_names[0]
    for i in range(len(unicode_names)-1):
        if unicode_names[i] != unicode_names[i+1]:
            return 'Other'
    return unicode_names[0]


def main():
    args = get_args()

    calculator = Calculator()

    # split. 指定分割词. eg: 唔需 -> 不 需要;
    if args.ignore_file is None:
        split = dict()
    else:
        split = dict()
        with open(args.split_file, 'r', 'utf-8') as f:
            for row in f:
                words = str(row).strip().split()
                if len(words) >= 2:
                    split[words[0]] = words[1:]

    if len(split) != 0 and not args.do_lowercase:
        new_split = dict()
        for k, words in split.items():
            new_split[k.lower()] = [w.lower() for w in words]
        split = new_split

    # ignore_words. 忽略词.
    if args.ignore_file is None:
        ignore_words = {'<blank>', '<unk>', '<sos/eos>'}
    else:
        ignore_words = set()
        with open(args.ignore_file, 'r', 'utf-8') as f:
            for row in f:  # line in unicode
                row = str(row).strip()
                if len(row) > 0:
                    ignore_words.add(row)

    if not args.do_lowercase:
        ignore_words = set([w.lower() for w in ignore_words])

    # padding symbol
    if args.padding_symbol == 'space':
        padding_symbol = ' '
    elif args.padding_symbol == 'underline':
        padding_symbol = '_'
    else:
        padding_symbol = ' '

    # eval
    default_clusters = defaultdict(dict)
    default_words = dict()
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        for row in f:
            row = json.loads(row)
            key: str = row['key']
            lab: str = row['txt']
            rec: str = row['hypothesis']

            if args.to_char:
                lab: List[str] = characterize(lab)
            else:
                lab: List[str] = lab.strip().split()

            if args.to_char:
                rec: List[str] = characterize(rec)
            else:
                rec: List[str] = rec.strip().split()

            if len(lab) == 0 or len(rec) == 0:
                continue

            lab = normalize(lab, ignore_words, args.do_lowercase, split)
            rec = normalize(rec, ignore_words, args.do_lowercase, split)

            for word in rec + lab:
                if word not in default_words:
                    default_cluster_name = default_cluster(word)

                    if word not in default_clusters[default_cluster_name]:
                        default_clusters[default_cluster_name][word] = 1

                    default_words[word] = default_cluster_name

            result = calculator.calculate(lab, rec)

            if args.verbose:
                if result['all'] != 0:
                    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
                else:
                    wer = 0.0
                print('WER: %4.2f %%' % wer, end=' ')
                print('N=%d C=%d S=%d D=%d I=%d' %
                      (result['all'], result['cor'], result['sub'], result['del'], result['ins']))

                space = {
                    'lab': list(),
                    'rec': list(),
                }
                for idx in range(len(result['lab'])):
                    len_lab = width(result['lab'][idx])
                    len_rec = width(result['rec'][idx])
                    length = max(len_lab, len_rec)
                    space['lab'].append(length-len_lab)
                    space['rec'].append(length-len_rec)
                upper_lab = len(result['lab'])
                upper_rec = len(result['rec'])
                lab1, rec1 = 0, 0
                while lab1 < upper_lab or rec1 < upper_rec:
                    if args.verbose > 1:
                        print('lab(%s):' % key, end=' ')
                    else:
                        print('lab:', end=' ')
                    lab2 = min(upper_lab, lab1 + args.max_words_per_line)
                    for idx in range(lab1, lab2):
                        token = result['lab'][idx]
                        print('{token}'.format(token=token), end='')
                        for n in range(space['lab'][idx]):
                            print(padding_symbol, end='')
                        print(' ', end='')
                    print()
                    if args.verbose > 1:
                        print('rec(%s):' % key, end=' ')
                    else:
                        print('rec:', end=' ')
                    rec2 = min(upper_rec, rec1 + args.max_words_per_line)
                    for idx in range(rec1, rec2):
                        token = result['rec'][idx]
                        print('{token}'.format(token=token), end='')
                        for n in range(space['rec'][idx]):
                            print(padding_symbol, end='')
                        print(' ', end='')
                    print('\n', end='\n')
                    lab1 = lab2
                    rec1 = rec2

        if args.verbose:
            print('===========================================================================')
            print()

        result = calculator.overall()
        if result['all'] != 0:
            wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
        else:
            wer = 0.0
        print('Overall -> %4.2f %%' % wer, end=' ')
        print('N=%d C=%d S=%d D=%d I=%d' %
              (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
        if not args.verbose:
            print()

        if args.verbose:
            for cluster_id in default_clusters:
                result = calculator.cluster([k for k in default_clusters[cluster_id]])
                if result['all'] != 0 :
                    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
                else:
                    wer = 0.0
                print('%s -> %4.2f %%' % (cluster_id, wer), end=' ')
                print('N=%d C=%d S=%d D=%d I=%d' %
                      (result['all'], result['cor'], result['sub'], result['del'], result['ins']))

            # compute separated WERs for word clusters
            if args.cluster_file is not None:
                cluster_id = ''
                cluster = []

                with open(args.cluster_file, 'r', encoding='utf-8') as f:
                    for row in f:
                        for token in row.rstrip('\n').split():
                            # end of cluster reached, like </Keyword>
                            if token[0:2] == '</' and token[len(token)-1] == '>' and \
                                    token.lstrip('</').rstrip('>') == cluster_id:
                                result = calculator.cluster(cluster)
                                if result['all'] != 0:
                                    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
                                else:
                                    wer = 0.0
                                print('%s -> %4.2f %%' % (cluster_id, wer), end=' ')
                                print('N=%d C=%d S=%d D=%d I=%d' %
                                      (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
                                cluster_id = ''
                                cluster = []
                            # begin of cluster reached, like <Keyword>
                            elif token[0] == '<' and token[len(token)-1] == '>' and \
                                    cluster_id == '':
                                cluster_id = token.lstrip('<').rstrip('>')
                                cluster = []
                            # general terms, like WEATHER / CAR / ...
                            else:
                                cluster.append(token)
            print()
            print('===========================================================================')

    return


if __name__ == '__main__':
    main()
