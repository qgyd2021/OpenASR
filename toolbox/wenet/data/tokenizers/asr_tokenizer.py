#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
from typing import List

from toolbox.wenet.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register('cjk_bpe_tokenizer')
class CjkBpeTokenizer(Tokenizer):
    """
    reference:
    https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/processor.py

    """
    def __init__(self,
                 bpe_model_file: str = None,
                 non_lang_symbols: List[str] = None,
                 split_with_space: bool = False,
                 ):
        self.bpe_model_file = bpe_model_file
        self.non_lang_symbols = non_lang_symbols or list()
        self.split_with_space = split_with_space

        if bpe_model_file is not None:
            import sentencepiece as spm
            bpe_model = spm.SentencePieceProcessor()
            bpe_model.load(bpe_model_file)
        else:
            bpe_model = None
        self.bpe_model = bpe_model

        self.non_lang_symbols_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})", flags=re.IGNORECASE)
        self.cjk_pattern = re.compile(r'([\u4e00-\u9fff])', flags=re.IGNORECASE)

    def _tokenize_by_bpe_model(self, text: str):
        tokens = []

        # "你好 ITS'S OKAY 的" -> ["你", "好", " ITS'S OKAY ", "的"]
        chars = self.cjk_pattern.split(text.upper())

        mix_chars = [w for w in chars if len(w.strip()) > 0]

        for ch_or_w in mix_chars:
            if self.cjk_pattern.fullmatch(ch_or_w) is not None:
                tokens.append(ch_or_w)
            else:
                for p in self.bpe_model.encode_as_pieces(ch_or_w):
                    tokens.append(p)

        return tokens

    def tokenize(self, text: str) -> List[str]:
        # split
        if len(self.non_lang_symbols) != 0:
            parts = self.non_lang_symbols_pattern.split(text)
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [text]

        # tokens
        tokens = []
        for part in parts:
            if part in self.non_lang_symbols:
                tokens.append(part)
                continue

            if self.bpe_model is not None:
                tokens.extend(self._tokenize_by_bpe_model(part))
                continue

            if self.split_with_space:
                part = part.split(' ')

            for ch in part:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)

        return tokens


def demo1():
    tokenizer = CjkBpeTokenizer(
        non_lang_symbols=[
            '<speech_noise>'
        ]
    )
    text = '不行<speech_noise>不行的ok西班牙语para que la reparacion del daño sea efectiva'
    result = tokenizer.tokenize(text)
    print(result)
    return


if __name__ == '__main__':
    demo1()
