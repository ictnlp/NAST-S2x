import os
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
imputer = load(
    "imputer_fn",
    sources=[
        os.path.join(module_path, "imputer.cpp"),
        os.path.join(module_path, "imputer.cu"),
        os.path.join(module_path, "best_alignment.cu"),
    ],
)


class ImputerLossFunction(Function):
    @staticmethod
    def forward(
        ctx,
        log_prob,
        targets,
        force_emits,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    ):
        input_lengths = input_lengths.to("cpu", dtype=torch.int64)
        target_lengths = target_lengths.to("cpu", dtype=torch.int64)

        loss, log_alpha = imputer.imputer_loss(
            log_prob,
            targets,
            force_emits,
            input_lengths,
            target_lengths,
            blank,
            zero_infinity,
        )

        ctx.save_for_backward(
            log_prob,
            targets,
            force_emits,
            input_lengths,
            target_lengths,
            loss,
            log_alpha,
        )
        ctx.blank = blank
        ctx.zero_infinity = zero_infinity

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        log_prob, targets, force_emits, input_lengths, target_lengths, loss, log_alpha = (
            ctx.saved_tensors
        )
        blank = ctx.blank
        zero_infinity = ctx.zero_infinity

        grad_input = imputer.imputer_loss_backward(
            grad_output,
            log_prob,
            targets,
            force_emits,
            input_lengths,
            target_lengths,
            loss,
            log_alpha,
            blank,
            zero_infinity,
        )

        return grad_input, None, None, None, None, None, None


imputer_loss_fn = ImputerLossFunction.apply


def imputer_loss(
    log_prob,
    targets,
    force_emits,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """The Imputer loss

    Parameters:
        log_prob (T, N, C): C = number of characters in alphabet including blank
                            T = input length
                            N = batch size
                            log probability of the outputs (e.g. torch.log_softmax of logits)
        targets (N, S): S = maximum number of characters in target sequences
        force_emits (N, T): sequence of ctc states that should be occur given times
                            that is, if force_emits is state s at time t, only ctc paths
                            that pass state s at time t will be enabled, and will be zero out the rest
                            this will be same as using cross entropy loss at time t
                            value should be in range [-1, 2 * S + 1), valid ctc states
                            -1 will means that it could be any states at time t (normal ctc paths)
        input_lengths (N): lengths of log_prob
        target_lengths (N): lengths of targets
        blank (int): index of blank tokens (default 0)
        reduction (str): reduction methods applied to the output. 'none' | 'mean' | 'sum'
        zero_infinity (bool): if true imputer loss will zero out infinities.
                              infinities mostly occur when it is impossible to generate
                              target sequences using input sequences
                              (e.g. input sequences are shorter than target sequences)
    """

    loss = imputer_loss_fn(
        log_prob,
        targets,
        force_emits,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    )

    input_lengths = input_lengths.to("cpu", dtype=torch.int64)
    target_lengths = target_lengths.to("cpu", dtype=torch.int64)

    if zero_infinity:
        inf = float("inf")
        loss = torch.where(loss == inf, loss.new_zeros(1), loss)

    if reduction == "mean":
        target_length = target_lengths.to(loss).clamp(min=1)

        return (loss / target_length).mean()

    elif reduction == "sum":
        return loss.sum()

    elif reduction == "none":
        return loss

    else:
        raise ValueError(
            f"Supported reduction modes are: mean, sum, none; got {reduction}"
        )


class ImputerLoss(nn.Module):
    def __init__(self, blank=0, reduction="mean", zero_infinity=False):
        """The Imputer loss

        Parameters:
            blank (int): index of blank tokens (default 0)
            reduction (str): reduction methods applied to the output. 'none' | 'mean' | 'sum'
            zero_infinity (bool): if true imputer loss will zero out infinities.
                                infinities mostly occur when it is impossible to generate
                                target sequences using input sequences
                                (e.g. input sequences are shorter than target sequences)

        Input:
            log_prob (T, N, C): C = number of characters in alphabet including blank
                                T = input length
                                N = batch size
                                log probability of the outputs (e.g. torch.log_softmax of logits)
            targets (N, S): S = maximum number of characters in target sequences
            force_emits (N, T): sequence of ctc states that should be occur given times
                            that is, if force_emits is state s at time t, only ctc paths
                            that pass state s at time t will be enabled, and will be zero out the rest
                            this will be same as using cross entropy loss at time t
                            value should be in range [-1, 2 * S + 1), valid ctc states
                            -1 will means that it could be any states at time t (normal ctc paths)
            input_lengths (N): lengths of log_prob
            target_lengths (N): lengths of targets"""
        super().__init__()

        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_prob, targets, force_emits, input_lengths, target_lengths):
        return imputer_loss(
            log_prob,
            targets,
            force_emits,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )


"""class ImputerLoss(nn.Module):
    def __init__(self, blank=0, reduction="mean", zero_infinity=False, mask_eps=1e-8):
        super().__init__()

        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.mask_eps = math.log(mask_eps)

    def forward(
        self, logit, targets_ctc, targets_ce, mask, input_lengths, targets_ctc_lengths
    ):
        n_target = logit.shape[-1]

        mask_e = mask.unsqueeze(-1)
        mask_exp = mask_e.repeat(1, 1, n_target)
        log_p_mask = logit.masked_fill(mask_exp == 1, self.mask_eps)
        mask_exp[:, :, self.blank] = 0
        log_p_mask = log_p_mask.masked_fill((mask_e == 1) & (mask_exp == 0), 0)
        log_p_mask = torch.log_softmax(log_p_mask, 2)

        ctc_loss = F.ctc_loss(
            log_p_mask.transpose(0, 1),
            targets_ctc,
            input_lengths,
            targets_ctc_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )

        ce_loss = F.cross_entropy(
            logit.view(-1, n_target), targets_ce.view(-1), reduction="none"
        )
        ce_loss = mask.view(-1) * ce_loss

        if self.reduction == "mean":
            ce_loss = ce_loss.mean()

        elif self.reduction == "sum":
            ce_loss = ce_loss.sum()

        return ctc_loss + ce_loss"""


def get_alignment_path(log_alpha, path):
    if log_alpha.shape[0] == 1:
        current_state = 0

    else:
        current_state = log_alpha[-2:, -1].argmax() + (log_alpha.shape[0] - 2)

    path_decode = [current_state]

    for t in range(path.shape[1] - 1, 0, -1):
        prev_state = path[current_state, t]
        path_decode.append(prev_state)
        current_state = prev_state

    return path_decode[::-1]


def ctc_decode(seq, blank=0):
    result = []

    prev = -1
    for s in seq:
        if s == blank:
            prev = s

            continue

        if prev == -1:
            result.append(s)

        else:
            if s != prev:
                result.append(s)

        prev = s

    return result


def best_alignment(
    log_prob, targets, input_lengths, target_lengths, blank=0, zero_infinity=False
):
    """Get best alignment (maximum probability sequence of ctc states)
       conditioned on log probabilities and target sequences

    Input:
        log_prob (T, N, C): C = number of characters in alphabet including blank
                            T = input length
                            N = batch size
                            log probability of the outputs (e.g. torch.log_softmax of logits)
        targets (N, S): S = maximum number of characters in target sequences
        input_lengths (N): lengths of log_prob
        target_lengths (N): lengths of targets
        blank (int): index of blank tokens (default 0)
        zero_infinity (bool): if true imputer loss will zero out infinities.
                            infinities mostly occur when it is impossible to generate
                            target sequences using input sequences
                            (e.g. input sequences are shorter than target sequences)

    Output:
        best_aligns (List[List[int]]): sequence of ctc states that have maximum probabilties
                                       given log probabilties, and compatible with target sequences"""
    nll, log_alpha, alignment = imputer.best_alignment(
        log_prob, targets, input_lengths, target_lengths, blank, zero_infinity
    )

    log_alpha = log_alpha.transpose(1, 2).detach().cpu().numpy()
    alignment = alignment.transpose(1, 2).detach().cpu().numpy()

    best_aligns = []

    for log_a, align, input_len, target_len in zip(
        log_alpha, alignment, input_lengths, target_lengths
    ):
        state_len = target_len * 2 + 1
        log_a = log_a[:state_len, :input_len]
        align = align[:state_len, :input_len]

        best_aligns.append(get_alignment_path(log_a, align))

    return best_aligns
