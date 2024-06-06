# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor

from dataclasses import dataclass, field

from .utilities import parse_anneal_argument, get_anneal_value

@dataclass
class LabelSmoothedDualImitationCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    label_smoothing_unit: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    use_ngram: bool = field(
        default=False,
    )
    use_ngram_unit: bool = field(
        default=False,
    )
    ngram_size: int = field(
        default=2,
        metadata={"help": "ngram order for NMLA training"}
    )
    glat_p: str = field(
        default="0",
    )
    glat_p_unit: str = field(
        default="0",
    )

@register_criterion("nat_loss_ngram_glat_s2u", dataclass=LabelSmoothedDualImitationCriterionConfig)
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, glat_p, glat_p_unit):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self._glat_p_anneal_params = parse_anneal_argument(glat_p)
        self._glat_p_unit_anneal_params = parse_anneal_argument(glat_p_unit)
        
        self.set_update_num(0)

    def set_update_num(self, update_num):
        self.glat_p = get_anneal_value(self._glat_p_anneal_params, update_num)
        self.glat_p_unit = get_anneal_value(self._glat_p_unit_anneal_params, update_num)
        
    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens_text"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target_text"], sample["prev_target"]
        transcript_tokens = sample["source_text"]
        tgt_units = sample["target_audio"]

        if sample.get("update_num", None) is not None: # in training            
            self.set_update_num(sample['update_num'])

        if max(self.glat_p, 0) == 0:
            glat = None
        else:
            glat = {
                "context_p": self.glat_p,
                "context_p_unit": self.glat_p_unit,
                "require_glance_grad": False
            }

        outputs, glat_info = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, transcript_tokens, tgt_units, glat)
        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if glat_info is not None:
            logging_output["glat_p"] = glat_info.get("glat_context_p", 0)
            logging_output["glat_p_unit"] = glat_info.get("glat_context_p_unit", 0)
            logging_output["glat_acc"] = glat_info.get("glat_acc", 0)
            logging_output["glat_acc_unit"] = glat_info.get("glat_acc_unit", 0)
            logging_output["glat_keep"] = glat_info.get("glat_keep", 0)
            logging_output["glat_keep_unit"] = glat_info.get("glat_keep_unit", 0)

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        glat_acc = utils.item(sum(log.get("glat_acc", 0) for log in logging_outputs))
        glat_keep = utils.item(sum(log.get("glat_keep", 0) for log in logging_outputs))
        glat_p = utils.item(sum(log.get("glat_p", 0) for log in logging_outputs))
        
        glat_acc_unit = utils.item(sum(log.get("glat_acc_unit", 0) for log in logging_outputs))
        glat_keep_unit = utils.item(sum(log.get("glat_keep_unit", 0) for log in logging_outputs))
        glat_p_unit = utils.item(sum(log.get("glat_p_unit", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        metrics.log_scalar(
            "glat_p", glat_p / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "glat_acc", glat_acc / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "glat_keep", glat_keep / sample_size, sample_size, round=3
        )
        
        metrics.log_scalar(
            "glat_p_unit", glat_p_unit / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "glat_acc_unit", glat_acc_unit / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "glat_keep_unit", glat_keep_unit / sample_size, sample_size, round=3
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
