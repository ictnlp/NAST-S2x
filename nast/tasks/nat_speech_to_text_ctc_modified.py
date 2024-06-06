# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging

from fairseq import utils, metrics
from fairseq.tasks import register_task
from nast.tasks.nat_speech_to_text import NATSpeechToTextTask
from nast.tasks.speech_to_text_modified import SpeechToTextModifiedTask

import math
import torch
import numpy as np

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@register_task("nat_speech_to_text_ctc_modified")
class NATSpeechToTextCTCModifiedTask(NATSpeechToTextTask):

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        
        blank_index = self.tgt_dict.add_symbol("<blank>")
        self.tgt_dict.blank_index = blank_index
        self.src_upsample_ratio = args.src_upsample_ratio
        self.main_context = args.main_context
        self.right_context = args.right_context
        self.unit_size = args.unit_size

    @classmethod
    def add_args(cls, parser):
        NATSpeechToTextTask.add_args(parser)
            
        parser.add_argument(
            "--main-context",
            type=int,
        )
        
        parser.add_argument(
            "--right-context",
            type=int,
        )
        
        
        parser.add_argument(
            '--unit-size',
            type=int
        )
        
   
    def _ctc_postprocess(self, tokens):
        _toks = tokens.int().tolist()
        deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
        hyp = tokens.new_tensor([v for v in deduplicated_toks if v != self.tgt_dict.blank_index])
        return hyp
    

    def reduce_metrics(self, logging_outputs, criterion):
        SpeechToTextModifiedTask.reduce_metrics(self, logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)    
    
    
    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe_tokenizer is not None:
                s = self.bpe_tokenizer.decode(s)
            if self.pre_tokenizer is not None:
                s = self.pre_tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = self._ctc_postprocess(gen_out[i][0]["tokens"])
            hyp = decode(hyp)
            ref = decode(
                utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
            hyps.append(hyp)
            refs.append(ref)

        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
        
    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        #max_positions = ((max_positions[0]-6) / self.src_upsample_ratio, max_positions[1])
        max_positions = (min(max_positions[0], max_positions[1] / self.src_upsample_ratio * self.unit_size * 4), max_positions[1])

        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions)
                )
            logger.warning(
                (
                    "{:,} samples have invalid sizes and will be skipped, "
                    "max_positions={}, first few sample ids={}"
                ).format(len(ignored), max_positions, ignored[:10])
            )

        filtered_indices = []
        upsample_filtered_cnt = 0
        short_filtered_cnt = 0
        
        for ind in indices:
            src_len, tgt_len = dataset.size(ind)
        
            length_prev = math.ceil(src_len / 2)
            length_prev = math.ceil(length_prev / 2)
            
            if length_prev == 0:
                short_filtered_cnt += 1
            elif dataset.is_train_split:   
                length_prev = length_prev // (self.unit_size) * self.src_upsample_ratio 

                if length_prev >= tgt_len + 2:
                    filtered_indices.append(ind)
                else:
                    upsample_filtered_cnt += 1

            else:
                filtered_indices.append(ind)
            

        logger.info(f"{short_filtered_cnt} samples have been filtered since too short")
        logger.info(f"{upsample_filtered_cnt} samples have been filtered since lamda * N < M")
        return filtered_indices

        # original_size = len(indices)
        # if ignore_invalid_inputs:
        #     max_positions = (
        #         (dataset.src_sizes[indices]).tolist(),
        #         (dataset.src_sizes[indices] * self.src_upsample_ratio).tolist(),
        #     )
        # indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        # if len(ignored) > 0:
        #     if not ignore_invalid_inputs:
        #         raise Exception(
        #             (
        #                 "Size of sample #{} is invalid (={}) since max_positions={}, "
        #                 "skip this example with --skip-invalid-size-inputs-valid-test"
        #             ).format(ignored[0], dataset.size(ignored[0]), max_positions)
        #         )

        #     logger.info(f"Dataset original size: {original_size}, filtered size: {len(indices)}")
        #return indices