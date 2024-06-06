# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import torch
import logging
import numpy as np
import math
from pathlib import Path

from argparse import Namespace
from fairseq import utils, metrics
from fairseq.tasks import register_task
from nast.tasks.speech_to_speech_modified import SpeechToSpeechModifiedTask

from nast.tasks.nat_speech_to_unit import NATSpeechToUnitTask


logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@register_task("nat_speech_to_unit_ctc_modified")
class NATSpeechToUnitCTCModifiedTask(NATSpeechToUnitTask):

    def __init__(self, args, tgt_dict, tgt_dict_unit):
        super().__init__(args, tgt_dict, tgt_dict_unit)
        text_blank_index = self.tgt_dict.add_symbol("<blank>")
        unit_blank_index = self.tgt_dict_unit.add_symbol("<blank>")
        
        self.tgt_dict.blank_index = text_blank_index
        self.tgt_dict_unit.blank_index = unit_blank_index
        
        self.main_context = args.main_context
        self.right_context = args.right_context
        
        self.unit_size = args.unit_size

    @classmethod
    def add_args(cls, parser):
        NATSpeechToUnitTask.add_args(parser)
        
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
    
    def _ctc_postprocess_unit(self, tokens):
        _toks = tokens.int().tolist()
        deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
        hyp = tokens.new_tensor([v for v in deduplicated_toks if v != self.tgt_dict_unit.blank_index])
        return hyp

    
    def valid_step(self, sample, model, criterion):
        model.eval()
        for task_name, task_obj in self.multitask_tasks.items():
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].eval()
            sample["multitask"][task_name]["prev_target"] = self.inject_noise(
                sample["multitask"][task_name]["target_text"],
            )
        sample["prev_target"] = self.inject_noise(sample["target_text"])

        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
            EVAL_BLEU_ORDER = 4
            if self.args.eval_bleu:
                bleu, bleu_unit = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                
                logging_output["_bleu_sys_len_unit"] = bleu_unit.sys_len
                logging_output["_bleu_ref_len_unit"] = bleu_unit.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
                    
                    logging_output["_bleu_counts_unit_" + str(i)] = bleu_unit.counts[i]
                    logging_output["_bleu_totals_unit_" + str(i)] = bleu_unit.totals[i]

        return loss, sample_size, logging_output    


    def reduce_metrics(self, logging_outputs, criterion):
        SpeechToSpeechModifiedTask.reduce_metrics(self, logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            counts_unit, totals_unit = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))
                
                counts_unit.append(sum_logs("_bleu_counts_unit_" + str(i)))
                totals_unit.append(sum_logs("_bleu_totals_unit_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))
                
                metrics.log_scalar("_bleu_counts_unit", np.array(counts_unit))
                metrics.log_scalar("_bleu_totals_unit", np.array(totals_unit))
                metrics.log_scalar("_bleu_sys_len_unit", sum_logs("_bleu_sys_len_unit"))
                metrics.log_scalar("_bleu_ref_len_unit", sum_logs("_bleu_ref_len_unit"))
                

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
                
                def compute_bleu_unit(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts_unit"].sum,
                        total=meters["_bleu_totals_unit"].sum,
                        sys_len=meters["_bleu_sys_len_unit"].sum,
                        ref_len=meters["_bleu_ref_len_unit"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)
                metrics.log_derived("bleu_unit", compute_bleu_unit)
  
    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode_unit(toks, escape_unk=False):
            s = self.tgt_dict_unit.string(
                toks.int().cpu(),
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            return s
        
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
        hyps_unit, refs_unit = [], []
        
        for i in range(len(gen_out)):
            hyp = self._ctc_postprocess(gen_out[i][0]["tokens"])
            hyp_unit = self._ctc_postprocess_unit(gen_out[i][0]["units"])
            
            hyp = decode(hyp)
            hyp_unit = decode_unit(hyp_unit)
            
            ref = decode(
                utils.strip_pad(sample["target_text"][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
            ref_unit = decode_unit(
                utils.strip_pad(sample["target_audio"][i], self.tgt_dict_unit.pad()),
                escape_unk=True,
            )
            
            hyps.append(hyp)
            hyps_unit.append(hyp_unit)
            
            refs.append(ref)
            refs_unit.append(ref_unit)

        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
            logger.info("example unit hypothesis: " + hyps_unit[0])
            logger.info("example unit reference: " + refs_unit[0])
            
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none"), sacrebleu.corpus_bleu(hyps_unit, [refs_unit], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs]), sacrebleu.corpus_bleu(hyps_unit, [refs_unit])
    
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

        
        max_positions = (min(max_positions[0], max_positions[1] / self.src_upsample_ratio * self.unit_size * 4, max_positions[2] / self.src_upsample_ratio / self.hidden_upsample_ratio * self.unit_size * 4), max_positions[1], max_positions[2])
        
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
        text_upsample_filtered_cnt = 0
        unit_upsample_filtered_cnt = 0
        total_upsample_filtered_cnt = 0
        
        for ind in indices:
            src_len, tgt_text_len, tgt_unit_len = dataset.size(ind)
            
            length_prev = math.ceil(src_len / 2)
            length_prev = math.ceil(length_prev / 2)

            if dataset.is_train_split:
                text_length_prev = math.ceil(length_prev / self.unit_size) * self.src_upsample_ratio 
                unit_length_prev = text_length_prev * self.hidden_upsample_ratio

                if text_length_prev < tgt_text_len + 2:
                    text_upsample_filtered_cnt += 1
                
                if unit_length_prev < tgt_unit_len + 2:
                    unit_upsample_filtered_cnt += 1
                
                if (text_length_prev >= tgt_text_len + 2) and (unit_length_prev >= tgt_unit_len + 2):
                    filtered_indices.append(ind)
                else:
                    total_upsample_filtered_cnt += 1
            else:
                filtered_indices.append(ind)      

        logger.info(f"{text_upsample_filtered_cnt} samples have been filtered since lamda * N < M (text)")
        logger.info(f"{unit_upsample_filtered_cnt} samples have been filtered since lamda * N < M (unit)")
        logger.info(f"{total_upsample_filtered_cnt} samples have been filtered since lamda * N < M (total)")
        
        return filtered_indices