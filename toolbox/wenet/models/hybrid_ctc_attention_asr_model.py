#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pad_sequence

from toolbox.wenet.common.util import sanitize
from toolbox.wenet.modules.encoders.encoder import Encoder
from toolbox.wenet.modules.decoders.decoder import Decoder
from toolbox.wenet.modules.loss import CTCLoss, LabelSmoothingLoss
from toolbox.wenet.utils.common import add_sos_eos, log_add, remove_duplicates_and_blank, \
    reverse_pad_list, th_accuracy, IGNORE_ID
from toolbox.wenet.utils.mask import mask_finished_preds, mask_finished_scores, make_pad_mask, subsequent_mask
from toolbox.wenet.models.model import Model


@Model.register('hybrid_ctc_attention_asr_model')
class HybridCtcAttentionAsrModel(Model):
    def __init__(self,
                 vocab_size: int,
                 encoder: Encoder,
                 decoder: Decoder,
                 ctc_loss: CTCLoss,
                 ctc_weight: float = 0.5,
                 att_loss: LabelSmoothingLoss = None,
                 ignore_id: int = IGNORE_ID,
                 reverse_weight: float = 0.0,
                 length_normalized_loss: bool = False,
                 ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.decoder = decoder
        self.ctc_loss = ctc_loss
        self.ctc_weight = ctc_weight
        self.att_loss = att_loss
        self.ignore_id = ignore_id
        self.reverse_weight = reverse_weight
        self.length_normalized_loss = length_normalized_loss

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        :param speech: shape=(batch_size, seq_len, ...).
        :param speech_lengths: shape=(batch_size,)
        :param text: shape=(batch_size,)
        :param text_lengths: shape=(batch_size,)
        :return:
        """
        # encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # ctc loss
        loss_ctc = torch.tensor(0.0)
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc_loss(
                encoder_out,
                encoder_out_lens,
                text,
                text_lengths
            )

        # attention loss
        loss_att = torch.tensor(0.0)
        acc_att = torch.tensor(0.0)
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(
                encoder_out,
                encoder_mask,
                text,
                text_lengths
            )

        # loss
        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        result = {
            'loss': loss,
            'loss_ctc': loss_ctc.detach(),
            'loss_att': loss_att.detach(),
            'acc_att': acc_att.detach(),
        }
        return result

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_mask: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(
            r_ys_pad, self.sos, self.eos, self.ignore_id)

        # forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask,
            ys_in_pad, ys_in_lens,
            r_ys_in_pad,
            self.reverse_weight
        )

        # compute attention loss
        loss_att = self.att_loss(decoder_out, ys_out_pad)

        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.att_loss(r_decoder_out, r_ys_out_pad)

        # attention loss
        loss_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight

        # accuracy
        acc_att = th_accuracy(
            pad_outputs=decoder_out.view(-1, self.vocab_size),
            pad_targets=ys_out_pad,
            ignore_label=self.ignore_id,
        )
        acc_att = torch.tensor(acc_att)
        return loss_att, acc_att

    def _forward_encoder(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )
        return encoder_out, encoder_mask

    def recognize(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_size: int = 10,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Apply beam search on attention decoder

        :param speech: torch.Tensor. shape=(batch_size, max_length, feat_dim).
        :param speech_lengths: torch.Tensor. shape=(batch_size,).
        :param beam_size: int. beam size for beam search.
        :param decoding_chunk_size: int. decoding chunk for dynamic chunk trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here.
        :param num_decoding_left_chunks:
        :param simulate_streaming: bool. whether do encoder forward in a streaming fashion
        :return:
        List[List[int]]. decoding result. shape=(batch_size, max_result_length).
        List[float]. scores.
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # (batch_size, max_length, encoder_dim)
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming
        )
        max_length = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size

        # (batch_size * beam_size, max_length, encoder_dim)
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, max_length, encoder_dim)
        # (batch_size * beam_size, 1, max_length)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1, max_length)

        # (batch_size * beam_size, 1)
        hyps = torch.ones([running_size, 1], dtype=torch.long, device=device).fill_(self.sos)
        # (beam_size,)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float)
        # (batch_size * beam_size, 1)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)

        # 2. Decoder forward step by step
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        for i in range(1, max_length + 1):
            if end_flag.sum() == running_size:
                break

            # 2.1 Forward decoder step
            # (batch_size * beam_size, i, i)
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(running_size, 1, 1).to(device)
            # logp: shape=(batch_size * beam_size, vocab)
            logp, cache = self.decoder.forward_one_step(encoder_out, encoder_mask, hyps, hyps_mask, cache)

            # 2.2. First beam prune: select topk best prob at current time
            # (batch_size * beam_size, beam_size)
            top_k_logp, top_k_index = logp.topk(beam_size)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)

            # 2.3. Second beam prune: select topk score with history
            # (batch_size * beam_size, beam_size)
            scores = scores + top_k_logp
            # (batch_size, beam_size * beam_size)
            scores = scores.view(batch_size, beam_size * beam_size)
            # scores: shape=(batch_size, beam_size)
            scores, offset_k_index = scores.topk(k=beam_size)
            # (batch_size * beam_size, 1)
            scores = scores.view(-1, 1)

            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (batch_size * beam_size * beam_size),
            # regard offset_k_index as (batch_size * beam_size),
            # then find offset_k_index in top_k_index
            # (batch_size, beam_size)
            base_k_index = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])
            base_k_index = base_k_index * beam_size * beam_size
            # (batch_size, beam_size)
            best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)

            # 2.5 Update best hyps
            # (batch_size, beam_size)
            best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)
            best_hyps_index = torch.div(best_k_index, beam_size, rounding_mode='floor')
            # (batch_size * beam_size, i)
            last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
            # (batch_size * beam_size, i + 1)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(batch_size, dtype=torch.long, device=device) * beam_size

        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]

        best_hyps = sanitize(best_hyps)
        best_scores = sanitize(best_scores)
        return best_hyps, best_scores

    def ctc_greedy_search(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Apply CTC greedy search

        :param speech: torch.Tensor. shape=(batch_size, max_length, feat_dim).
        :param speech_lengths: torch.Tensor. shape=(batch_size,).
        :param decoding_chunk_size: int. decoding chunk for dynamic chunk trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
        :param num_decoding_left_chunks:
        :param simulate_streaming: bool. whether do encoder forward in a streaming fashion.
        :return:
        List[List[int]]. best path result.
        List[float]. scores.
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]

        # (batch_size, max_length, encoder_dim)
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming
        )
        max_length = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # (batch_size, max_length, vocab_size)
        ctc_probs = self.ctc_loss.log_softmax(encoder_out)

        # (batch_size, max_length, 1)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)
        # (batch_size, max_length)
        topk_index = topk_index.view(batch_size, max_length)
        # (batch_size, max_length)
        mask = make_pad_mask(encoder_out_lens, max_length)
        # (batch_size, max_length)
        topk_index = topk_index.masked_fill_(mask, self.eos)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores, _ = topk_prob.max(1)
        scores = scores.squeeze(-1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]

        return hyps, scores

    def _ctc_prefix_beam_search(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_size: int = 10,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ) -> Tuple[Tuple[List[List[int]], List[float]], torch.Tensor]:
        """
        CTC prefix beam search inner implementation
        :param speech: torch.Tensor. shape=(batch_size, max_length, feat_dim).
        :param speech_lengths: torch.Tensor. shape=(batch_size,).
        :param beam_size: int, beam size for beam search.
        :param decoding_chunk_size: int. decoding chunk for dynamic chunk trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
        :param num_decoding_left_chunks:
        :param simulate_streaming: bool. whether do encoder forward in a streaming fashion.
        :return:
        Tuple[List[List[int]], List[float]]. n best results
        torch.Tensor. encoder output, (1, max_len, encoder_dim),
                        it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # for CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1

        # 1. encoder forward and get CTC score
        # (batch_size, max_length, encoder_dim)
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)
        max_length = encoder_out.size(1)

        # (1, max_length, vocab_size)
        ctc_probs = self.ctc_loss.log_softmax(encoder_out)
        ctc_probs = ctc_probs.squeeze(0)

        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]

        # 2. CTC beam search step by step
        for t in range(0, max_length):
            # (vocab_size,)
            log_p = ctc_probs[t]

            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))

            # 2.1 first beam prune: select top k best
            # (beam_size,)
            top_k_log_p, top_k_index = log_p.topk(beam_size)
            for s in top_k_index:
                s = s.item()
                ps = log_p[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_size: int = 10,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        apply CTC prefix beam search.
        :param speech: torch.Tensor. shape=(batch_size, max_length, feat_dim).
        :param speech_lengths: torch.Tensor. shape=(batch_size,).
        :param beam_size: int, beam size for beam search.
        :param decoding_chunk_size: int. decoding chunk for dynamic chunk trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
        :param num_decoding_left_chunks:
        :param simulate_streaming: bool. whether do encoder forward in a streaming fashion.
        :return:
        List[List[int]]: CTC prefix beam search n best results.
        List[float]. scores.
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming)

        hyp, score = hyps[0]
        hyps = sanitize([hyp])
        scores = sanitize([score])
        return hyps, scores

    def attention_rescoring(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_size: int = 10,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            ctc_weight: float = 0.0,
            simulate_streaming: bool = False,
            reverse_weight: float = 0.0,
    ) -> List[int]:
        """
        apply attention rescoring decoding, CTC prefix beam search
        is applied first to get n best, then we resoring the n best on
        attention decoder with corresponding encoder out
        :param speech: torch.Tensor. shape=(batch_size, max_length, feat_dim).
        :param speech_lengths: torch.Tensor. shape=(batch_size,).
        :param beam_size: int, beam size for beam search.
        :param decoding_chunk_size: int. decoding chunk for dynamic chunk trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
        :param num_decoding_left_chunks:
        :param ctc_weight: float, ctc score weight
        :param simulate_streaming: bool. whether do encoder forward in a streaming fashion.
        :param reverse_weight: float, right to left decoder weight
        :return:
        List[List[int]]: Attention rescoring result
        List[float]. scores.
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # for attention rescoring we only support batch_size=1
        assert batch_size == 1

        # encoder_out: (1, max_length, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)

        assert len(hyps) == beam_size

        # (beam_size, max_hyps_len)
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)
        ori_hyps_pad = hyps_pad

        # (beam_size,)
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)

        # Add <sos> at begining
        hyps_lens = hyps_lens + 1
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)

        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos, self.ignore_id)

        # (beam_size, max_hyps_len, vocab_size)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.detach().cpu().numpy()

        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()

        # only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i

        hyps = sanitize([hyps[best_index][0]])
        scores = sanitize([best_score])
        return hyps, scores


def demo1():
    from toolbox.wenet.modules.encoders.bilstm_encoder import BiLstmEncoder
    from toolbox.wenet.modules.decoders.bilstm_decoder import AttentionBiLstmDecoder
    from toolbox.wenet.modules.loss import CTCLoss, LabelSmoothingLoss
    from toolbox.wenet.modules.subsampling import Conv2dSubsampling4
    from toolbox.wenet.modules.embedding import SinusoidalPositionalEncoding

    vocab_size = 10
    num_mel_bins = 80
    batch_size = 2
    max_seq_len_in = 100
    encoder_hidden_size = 128
    encoder_output_size = 128
    encoder_num_layers = 2

    decoder_hidden_size = 128
    decoder_output_size = 128
    decoder_num_layers = 2

    xs = torch.randn(size=(batch_size, max_seq_len_in, num_mel_bins))
    xs_lens = torch.randint(int(max_seq_len_in * 0.6), max_seq_len_in, size=(batch_size,), dtype=torch.long)
    xs_lens[0] = max_seq_len_in

    ys_in_pad = torch.tensor(data=[[1, 2, 1, 0], [3, 2, 1, 1]], dtype=torch.long)
    ys_in_lens = torch.tensor(data=[3, 4], dtype=torch.long)

    asr_model = HybridCtcAttentionAsrModel(
        vocab_size=vocab_size,
        encoder=BiLstmEncoder(
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            output_size=encoder_output_size,
            subsampling=Conv2dSubsampling4(
                input_dim=num_mel_bins,
                output_dim=encoder_hidden_size,
                dropout_rate=0.1,
                positional_encoding=SinusoidalPositionalEncoding(
                    embedding_dim=encoder_hidden_size,
                    dropout_rate=0.1,
                )
            ),
        ),
        decoder=AttentionBiLstmDecoder(
            vocab_size=vocab_size,
            input_size=encoder_output_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
        ),
        ctc_loss=CTCLoss(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
        ),
        att_loss=LabelSmoothingLoss(
            vocab_size=vocab_size,
            padding_idx=IGNORE_ID,
            smoothing=0.1,
        ),
    )

    result = asr_model.forward(speech=xs, speech_lengths=xs_lens, text=ys_in_pad, text_lengths=ys_in_lens)
    print(result)
    return


def demo2():
    from toolbox.wenet.modules.encoders.transformer_encoder import ConformerEncoder
    from toolbox.wenet.modules.decoders.transformer_decoder import TransformerDecoder
    from toolbox.wenet.modules.loss import CTCLoss, LabelSmoothingLoss
    from toolbox.wenet.modules.subsampling import Conv2dSubsampling4
    from toolbox.wenet.modules.embedding import SinusoidalPositionalEncoding

    vocab_size = 10
    num_mel_bins = 80
    batch_size = 2
    max_seq_len_in = 1000
    encoder_hidden_size = 128
    encoder_output_size = 128
    encoder_num_layers = 2
    encoder_attention_heads = 2
    encoder_linear_units = 128

    decoder_num_layers = 2
    decoder_attention_heads = 2
    decoder_linear_units = 128

    xs = torch.randn(size=(batch_size, max_seq_len_in, num_mel_bins))
    xs_lens = torch.randint(int(max_seq_len_in * 0.6), max_seq_len_in, size=(batch_size,), dtype=torch.long)
    xs_lens[0] = max_seq_len_in

    ys_in_pad = torch.tensor(data=[[1, 2, 1, 0], [3, 2, 1, 1]], dtype=torch.long)
    ys_in_lens = torch.tensor(data=[3, 4], dtype=torch.long)

    asr_model = HybridCtcAttentionAsrModel(
        vocab_size=vocab_size,
        encoder=ConformerEncoder(
            subsampling=Conv2dSubsampling4(
                input_dim=num_mel_bins,
                output_dim=encoder_hidden_size,
                dropout_rate=0.1,
                positional_encoding=SinusoidalPositionalEncoding(
                    embedding_dim=encoder_hidden_size,
                    dropout_rate=0.1,
                )
            ),
            output_size=encoder_hidden_size,
            attention_heads=encoder_attention_heads,
            linear_units=encoder_linear_units,
            num_blocks=encoder_num_layers,
            use_dynamic_chunk=True,
        ),
        decoder=TransformerDecoder(
            vocab_size=vocab_size,
            input_size=encoder_output_size,
            attention_heads=decoder_attention_heads,
            linear_units=decoder_linear_units,
            num_blocks=decoder_num_layers,
        ),
        ctc_loss=CTCLoss(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
        ),
        att_loss=LabelSmoothingLoss(
            vocab_size=vocab_size,
            padding_idx=IGNORE_ID,
            smoothing=0.1,
        ),
    )

    result = asr_model.forward(speech=xs, speech_lengths=xs_lens, text=ys_in_pad, text_lengths=ys_in_lens)
    print(result)
    return


if __name__ == '__main__':
    demo1()
    demo2()
