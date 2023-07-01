#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import deque, namedtuple
from typing import Tuple, Generator

import librosa
import numpy as np
from scipy.io import wavfile
import wave
import webrtcvad

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        default=(project_path / 'datasets/1f991410-14e6-4017-9237-62df3098d2d2-user.wav').as_posix(),
                        type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--vad_agg_mode', default=3, type=int, help='The level of aggressiveness of the VAD: [0-3]')

    parser.add_argument('--resample_rate', default=8000, type=str)
    parser.add_argument('--mono', default=True, type=bool)

    args = parser.parse_args()

    return args


Frame = namedtuple('Frame', ['bytes', 'timestamp', 'duration'])


class FairseqVad(object):
    """
    reference:
    https://github.com/facebookresearch/fairseq/blob/main/examples/speech_synthesis/preprocessing/vad/__init__.py
    """
    def __init__(self,
                 frame_duration: float = 0.03,
                 scale: float = 6e-5,
                 threshold: float = 0.3,
                 vad_agg_mode: int = 3
                 ):
        self.frame_duration = frame_duration
        # 1 / (sample_rate * 2)
        self.scale = scale
        self.threshold = threshold

        self.vad = webrtcvad.Vad(vad_agg_mode)

    @staticmethod
    def read_wav_to_pcm_data(filename: str) -> Tuple[bytes, int]:
        """

        should be monophonic audio recording.
        data type should be int16.

        :param filename:
        :return:
        """
        with wave.open(filename, 'rb') as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2

            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data: bytes = wf.readframes(wf.getnframes())

        return pcm_data, sample_rate

    @staticmethod
    def write_pcm_data_to_wav(filename: str, pcm_data: bytes, sample_rate: int) -> str:
        """
        :param filename:
        :param pcm_data:
        :param sample_rate:
        :return:
        """
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return filename

    def pcm_data_to_frames(self, pcm_data: bytes, sample_rate: int = 8000) -> Generator[Frame, None, None]:
        """
        :param pcm_data:
        :param sample_rate:
        :return:
        """
        n = int(sample_rate * self.frame_duration * 2)

        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(pcm_data):
            yield Frame(pcm_data[offset:offset + n], timestamp, duration)

            timestamp += duration
            offset += n

    def vad_collector(self,
                      frames: Generator[Frame, None, None],
                      sample_rate: int,
                      padding_duration: float = 0.3,
                      ):
        """
        Filters out non-voiced audio frames.
        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.
        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.
        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.
        :param frames: a source of audio frames (sequence or generator).
        :param sample_rate: int, the audio sample rate, in Hz.
        :param padding_duration: float, the amount to pad the window.
        :return: a generator that yields PCM audio data.
        """
        num_padding_frames = int(padding_duration / self.frame_duration)

        # we use a deque for our sliding window/ring buffer.
        ring_buffer = deque(maxlen=num_padding_frames)

        # we have two states: TRIGGERED and NOT_TRIGGERED. we start in the NOT_TRIGGERED state.
        triggered = False

        voiced_frames = list()
        for frame in frames:
            is_speech = self.vad.is_speech(frame.bytes, sample_rate)

            #
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])

                # If we're NOT_TRIGGERED and more than 90% of the frames in the ring buffer are voiced frames,
                # then enter the TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    # We want to yield all the audio we see from now until we are NOT_TRIGGERED,
                    # but we have to start with the audio that's already in the ring buffer.
                    for f, _ in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])

                # If more than 90% of the frames in the ring buffer are unvoiced,
                # then enter NOT_TRIGGERED and yield whatever audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield [
                        b''.join([f.bytes for f in voiced_frames]),
                        voiced_frames[0].timestamp,
                        voiced_frames[-1].timestamp
                    ]
                    ring_buffer.clear()
                    voiced_frames = list()

        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        if voiced_frames:
            yield [
                b''.join([f.bytes for f in voiced_frames]),
                voiced_frames[0].timestamp,
                voiced_frames[-1].timestamp
            ]

    def search_voice_activity_by_filename(self, filename: str):
        pcm_data, sample_rate = self.read_wav_to_pcm_data(filename)

        frames = self.pcm_data_to_frames(pcm_data, sample_rate)

        segments = self.vad_collector(
            frames=frames,
            sample_rate=sample_rate,
        )
        result = list()
        timestamp_start = 0.0
        timestamp_end = 0.0
        for i, segment in enumerate(segments):
            if i and timestamp_start:
                sil_duration = segment[1] - timestamp_end
                if sil_duration > self.threshold:
                    result.append((timestamp_start, timestamp_end))
                    timestamp_start = segment[1]
                    timestamp_end = segment[2]
                else:
                    timestamp_end = segment[2]
            else:
                timestamp_start = segment[1]
                timestamp_end = segment[2]

        if timestamp_end > timestamp_start:
            result.append((timestamp_start, timestamp_end))
        return result

    def remove_silence_by_filename(self, filename: str, output_filename: str):
        pcm_data, sample_rate = self.read_wav_to_pcm_data(filename)

        frames = self.pcm_data_to_frames(pcm_data, sample_rate)

        segments = self.vad_collector(
            frames=frames,
            sample_rate=sample_rate,
        )
        merge_segments = list()
        timestamp_start = 0.0
        timestamp_end = 0.0
        # removing start, end, and long sequences of sils
        for i, segment in enumerate(segments):
            merge_segments.append(segment[0])
            if i and timestamp_start:
                sil_duration = segment[1] - timestamp_end
                if sil_duration > self.threshold:
                    merge_segments.append(int(self.threshold / self.scale) * b'\x00')
                else:
                    merge_segments.append(int((sil_duration / self.scale)) * b'\x00')
            timestamp_start = segment[1]
            timestamp_end = segment[2]

        segment = b''.join(merge_segments)
        self.write_pcm_data_to_wav(output_filename, segment, sample_rate)
        return output_filename


def main():
    args = get_args()

    f_vad = FairseqVad(
        scale=1 / (8000 * 2),
        vad_agg_mode=args.vad_agg_mode
    )

    f_vad.remove_silence_by_filename(
        filename=args.filename,
        output_filename='temp.wav'
    )

    voice_activity_list = f_vad.search_voice_activity_by_filename(
        filename=args.filename,
    )
    print(voice_activity_list)

    return


if __name__ == '__main__':
    main()
