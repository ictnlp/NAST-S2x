# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import torch
import logging
import numpy as np

from argparse import Namespace
from fairseq import utils, metrics
from fairseq.tasks import register_task
from fairseq.utils import new_arange
from fairseq.optim.amp_optimizer import AMPOptimizer

from nast.tasks.speech_to_text_modified import SpeechToTextModifiedTask

from nast.datasets.nat_speech_to_text_dataset import (
    NATSpeechToTextDataset,
    NATSpeechToTextDatasetCreator,
)


logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@register_task("nat_speech_to_text")
class NATSpeechToTextTask(SpeechToTextModifiedTask):

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.pre_tokenizer = self.build_tokenizer(self.args)
        self.bpe_tokenizer = self.build_bpe(self.args)

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        if self.args.eval_bleu:
            gen_args = json.loads(self.args.eval_bleu_args)
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    @classmethod
    def add_args(cls, parser):
        SpeechToTextModifiedTask.add_args(parser)
        parser.add_argument(
            "--noise",
            type=str,
            default="random_delete",
            choices=["random_delete", "random_mask", "no_noise", "full_mask"],
            help="type of noise",
        )
        parser.add_argument(
            "--src-upsample-ratio",
            type=int,
            default=None,
            help="Specify the graph size with a upsample factor (lambda).  Graph Size = \\lambda * src_length",
        )
        # options for reporting BLEU during validation
        parser.add_argument(
            "--eval-bleu",
            action="store_true",
            help="evaluation with BLEU scores",
        )
        parser.add_argument(
            "--eval-bleu-detok",
            type=str,
            default="space",
            help="detokenize before computing BLEU (e.g., 'moses'); "
                 "required if using --eval-bleu; use 'space' to "
                 "disable detokenization; see fairseq.data.encoders "
                 "for other options",
        )
        parser.add_argument(
            "--eval-bleu-detok-args",
            type=str,
            metavar="JSON",
            help="args for building the tokenizer, if needed",
        )
        parser.add_argument(
            "--eval-tokenized-bleu",
            action="store_true",
            default=False,
            help="compute tokenized BLEU instead of sacrebleu",
        )
        parser.add_argument(
            "--eval-bleu-remove-bpe",
            nargs="?",
            const="@@ ",
            default=None,
            help="remove BPE before computing BLEU",
        )
        parser.add_argument(
            "--eval-bleu-args",
            type=str,
            metavar="JSON",
            help="generation args for BLUE scoring, "
                 "e.g., '{\"beam\": 4, \"lenpen\": 0.6}'",
        )
        parser.add_argument(
            "--eval-bleu-print-samples",
            action="store_true",
            help="print sample generations during validation",
        )
        parser.add_argument(
            "--eval-bleu-bpe",
            type=str,
            metavar="BPE",
            default=None,
            help="args for building the bpe, if needed",
        )
        parser.add_argument(
            "--eval-bleu-bpe-path",
            type=str,
            metavar='BPE',
            help="args for building the bpe, if needed",
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = NATSpeechToTextDatasetCreator.from_tsv(
            root=self.args.data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
            multitask=self.multitask_tasks,
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.args.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.args.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.args.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from ..generators.s2t_nat_generator import S2TNATGenerator

        return S2TNATGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 0),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs): 
        return NATSpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):  
        model.train()      
        for task_name, task_obj in self.multitask_tasks.items():
            criterion.set_multitask_loss_weight(
                task_name, task_obj.args.get_loss_weight(update_num)
            )
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].train()
            sample["multitask"][task_name]["prev_target"] = self.inject_noise(
                sample["multitask"][task_name]["target"],
            )
        sample["prev_target"] = self.inject_noise(sample["target"])
        sample["update_num"] = update_num

        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        for task_name, task_obj in self.multitask_tasks.items():
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].eval()
            sample["multitask"][task_name]["prev_target"] = self.inject_noise(
                sample["multitask"][task_name]["target"],
            )
        sample["prev_target"] = self.inject_noise(sample["target"])

        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
            EVAL_BLEU_ORDER = 4
            if self.args.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

        return loss, sample_size, logging_output
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                if key in logging_outputs[0]:
                    return sum(log[key].cpu().numpy() for log in logging_outputs)
                return sum(log.get(key, 0) for log in logging_outputs)

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
            hyp = decode(gen_out[i][0]["tokens"])
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
        filtered_cnt = 0
        for ind in indices:
            src_len, tgt_len = dataset.size(ind)
            try:
                if src_len * self.args.src_upsample_ratio >= tgt_len + 2:
                    filtered_indices.append(ind)
                else:
                    filtered_cnt += 1
            except:
                filtered_indices.append(ind)

        logger.info(f"{filtered_cnt} samples have been filtered since lamda * N < M")

        return filtered_indices