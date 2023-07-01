#!/usr/bin/python3
# -*- coding: utf-8 -*-
import platform
import random
import re
from typing import Dict, List, Union

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from toolbox.wenet.common.registrable import Registrable
from toolbox.wenet.data.tokenizers.asr_tokenizer import CjkBpeTokenizer
from toolbox.wenet.data.vocabulary import Vocabulary


class Preprocess(Registrable):
    def __init__(self, work_on_eval: bool = True):
        self.work_on_eval = work_on_eval

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        """
        return None if you wish to discard the sample.
        """
        raise NotImplementedError


@Preprocess.register('load_wav')
class LoadWav(Preprocess):
    def __init__(self):
        super().__init__()

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        """
        :param sample: {key, wav, txt}
        :return: sample: {key, waveform, txt, sample_rate}
        """
        if sample is None:
            return None

        key = sample['key']
        wav_file = sample['wav']
        txt = sample['txt']

        waveform, sample_rate = torchaudio.load(wav_file)

        result = {
            'key': key,
            'waveform': waveform,
            'txt': txt,
            'sample_rate': sample_rate,
        }
        return result


@Preprocess.register('resample')
class Resample(Preprocess):
    def __init__(self,
                 resample_rate: int = 16000
                 ):
        super().__init__()
        self.resample_rate = resample_rate

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        """
        :param sample: {key, waveform, txt, sample_rate}
        :return: sample: {key, waveform, txt, sample_rate}
        """
        if sample is None:
            return None

        sample_rate = sample['sample_rate']
        waveform = sample['waveform']

        new_wav = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=self.resample_rate
        )(waveform)

        sample['waveform'] = new_wav
        sample['sample_rate'] = self.resample_rate

        return sample


@Preprocess.register('speed_perturb')
class SpeedPerturb(Preprocess):
    """
    requires sox
    """
    def __init__(self,
                 speeds: List[float] = None
                 ):
        super().__init__(work_on_eval=False)
        self.speeds = speeds or [0.9, 1.0, 1.1]

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        """
        :param sample: {key, waveform, sample_rate}
        :return: sample: {key, waveform, sample_rate}
        """
        if sample is None:
            return None

        if platform.system() == 'Windows':
            return sample

        assert 'sample_rate' in sample
        assert 'waveform' in sample

        sample_rate = sample['sample_rate']
        waveform = sample['waveform']
        speed = random.choice(self.speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                tensor=waveform,
                sample_rate=sample_rate,
                effects=[['speed', str(speed)], ['rate', str(sample_rate)]]
            )
            sample['waveform'] = wav
        return sample


@Preprocess.register('cjk_bpe_tokenize')
class CjkBpeTokenize(Preprocess):
    def __init__(self,
                 bpe_model_file: str = None,
                 non_lang_symbols: List[str] = None,
                 split_with_space: bool = False,
                 ):
        super().__init__()
        self.tokenizer = CjkBpeTokenizer(
            bpe_model_file=bpe_model_file,
            non_lang_symbols=non_lang_symbols,
            split_with_space=split_with_space,
        )

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        """
        :param sample: {key, waveform, txt, sample_rate}
        :return: sample: {key, waveform, txt, sample_rate, token_list}
        """
        if sample is None:
            return None

        txt = sample['txt']
        token_list = self.tokenizer.tokenize(txt)
        sample['token_list'] = token_list
        return sample


@Preprocess.register('map_tokens_to_ids')
class MapTokensToIds(Preprocess):
    def __init__(self,
                 vocabulary: Vocabulary,
                 namespace: str = 'tokens'
                 ):
        super().__init__()
        self.vocabulary = vocabulary
        self.namespace = namespace

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        """
        :param sample: {key, ..., token_list}
        :return: sample: {key, ..., token_list, index_list}
        """
        if sample is None:
            return None

        token_list = sample['token_list']

        index_list = list()
        for token in token_list:
            index = self.vocabulary.get_token_index(token, namespace=self.namespace)
            index_list.append(index)

        sample['index_list'] = torch.tensor(data=index_list, dtype=torch.long)
        return sample


@Preprocess.register('waveform_to_fbank')
class WaveFormToFbank(Preprocess):
    def __init__(self,
                 num_mel_bins: int = 23,
                 frame_length: int = 25,
                 frame_shift: int = 10,
                 dither: float = 0.0
                 ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        """
        :param sample: {key, waveform, sample_rate}
        :return: sample: {key, feat, sample_rate}
        """
        if sample is None:
            return None

        sample_rate = sample['sample_rate']
        waveform = sample['waveform']
        waveform = waveform * (1 << 15)

        # (m, ``num_mel_bins + use_energy``)
        feat = kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=self.dither,
            energy_floor=0.0,
            sample_frequency=sample_rate
        )

        sample.pop('waveform')
        sample['feat'] = feat
        return sample


@Preprocess.register('spectrum_aug')
class SpectrumAug(Preprocess):
    def __init__(self,
                 num_t_mask: int = 2,
                 num_f_mask: int = 2,
                 max_t: int = 50,
                 max_f: int = 10,
                 max_w: int = 80
                 ):
        super().__init__(work_on_eval=False)
        self.num_t_mask = num_t_mask
        self.num_f_mask = num_f_mask
        self.max_t = max_t
        self.max_f = max_f
        self.max_w = max_w

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        if sample is None:
            return None

        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(self.num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(self.num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, self.max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        return sample


@Preprocess.register('spectrum_substitute')
class SpectrumSubstitute(Preprocess):
    def __init__(self,
                 max_t: int = 20,
                 num_t_sub: int = 3
                 ):
        super().__init__(work_on_eval=False)
        self.max_t = max_t
        self.num_t_sub = num_t_sub

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        """
        :param sample: {key, feat, sample_rate}
        :return: sample: {key, feat, sample_rate}
        """
        if sample is None:
            return None

        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)

        for i in range(self.num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)

            y[start:end, :] = x[start - pos:end - pos, :]

        sample['feat'] = y
        return sample


@Preprocess.register('spectrum_trim')
class SpectrumTrim(Preprocess):
    """
    https://arxiv.org/abs/2211.00522
    """

    def __init__(self, max_t: int = 20):
        super().__init__(work_on_eval=False)
        self.max_t = max_t

    def process(self, sample: Union[dict, None]) -> Union[dict, None]:
        if sample is None:
            return None

        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        max_frames = x.size(0)
        length = random.randint(1, self.max_t)
        if length < max_frames / 2:
            y = x.clone().detach()[:max_frames - length]
            sample['feat'] = y
        return sample
