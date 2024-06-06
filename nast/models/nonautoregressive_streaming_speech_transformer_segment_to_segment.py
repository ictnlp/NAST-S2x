# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from nast.generators.s2t_nat_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from fairseq import checkpoint_utils

from typing import Union, Optional, Dict, List, Any
from nast.models.torch_imputer import best_alignment
from collections import Counter
import math
import logging
logger = logging.getLogger(__name__)

from pathlib import Path

from ..modules.unidirectional_encoder import UnidirectionalAudioTransformerEncoder


DEFAULT_MAX_TEXT_POSITIONS = 1024
DEFAULT_MAX_AUDIO_POSITIONS = 6000



def _mean_pooling_in_block(enc_feats, src_masks, block_size, num_block):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    
    T, B, C = enc_feats.size()

    dim_t_left = T % block_size

    if dim_t_left is not 0:
        tail = enc_feats.new(block_size - dim_t_left, B, C).fill_(0)
        enc_feats = torch.cat([enc_feats, tail], dim=0).contiguous()
        mask_tail = src_masks.new(B, block_size - dim_t_left).fill_(0)
        src_masks = torch.cat([src_masks, mask_tail], dim=1).contiguous()

    enc_feats_block = enc_feats[:(num_block * block_size)].view(num_block, block_size, B, C).transpose(0, 1) # block_size, num_block, B, C  
    
    
    src_masks_block = src_masks[:, :(num_block * block_size)].view(B, num_block, block_size).transpose(0, 2) # block_size, num_block, B
    src_masks_float = src_masks_block.type_as(enc_feats)
    pooled_src_masks = src_masks_block.any(dim = 0).transpose(0,1) # B, num_block
    non_zero_masks = src_masks_block.any(dim = 0) # num_block, B
    enc_feats_temp = (
        (enc_feats_block / src_masks_float.sum(dim=0)[None, :, :, None]) * src_masks_float[:, :, :, None]
    )
    pooled_enc_feats = torch.where(non_zero_masks.unsqueeze(0).unsqueeze(-1), enc_feats_temp, enc_feats_block).sum(0) # num_block, B, C 
    pooled_enc_feats = pooled_enc_feats.transpose(0,1)
    
    return pooled_enc_feats, pooled_src_masks



def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    max_src_len = src_lens.max()
    bsz = src_lens.size(0)
    ratio = int(max_trg_len / max_src_len)
    index_t = utils.new_arange(trg_lens, max_src_len)
    index_t = torch.repeat_interleave(index_t, repeats=ratio, dim=-1).unsqueeze(0).expand(bsz, -1)
    return index_t 


@register_model("nonautoregressive_streaming_speech_transformer_segment_to_segment")
class NATransformerModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.src_upsample_ratio = args.src_upsample_ratio
        self.unit_size = args.unit_size
    
    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            '--rand-pos-encoder',
            type=int
        )
        parser.add_argument(
            '--conv-type',
            type=str
        )
        parser.add_argument(
            '--no-audio-positional-embeddings',
            type=bool
        )
        parser.add_argument(
            '--encoder-max-relative-position',
            type=int
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
    

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        
        
        if getattr(args, "max_audio_positions", None) is None:
            args.max_audio_positions = DEFAULT_MAX_AUDIO_POSITIONS
        if getattr(args, "max_text_positions", None) is None:
            args.max_text_positions = DEFAULT_MAX_TEXT_POSITIONS
        
        args.max_source_positions = args.max_audio_positions
        args.max_target_positions = args.max_text_positions
        


        decoder_embed_tokens = cls.build_embedding(
            args, task.target_dictionary, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task.target_dictionary, decoder_embed_tokens, encoder.main_context)

        model = cls(args, encoder, decoder)
        
        return model

    @classmethod
    def build_encoder(cls, args):
        encoder = UnidirectionalAudioTransformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")

        return encoder
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, main_context):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens, main_context)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    
    def sequence_ngram_loss_with_logits(self, logits, logit_mask, targets, blank_index):
        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        
        
        if self.args.ngram_size == 1:
            loss = self.compute_ctc_1gram_loss(log_probs, logit_mask, targets, blank_index)
        elif self.args.ngram_size == 2:
            loss = self.compute_ctc_bigram_loss(log_probs, logit_mask, targets, blank_index)
        else:
            raise NotImplementedError
        
        
        return loss
    
    
    def compute_ctc_1gram_loss(self, log_probs, logit_mask, targets, blank_index):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)
        bow = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        bow[:,blank_index] = 0
        ref_bow = torch.zeros(batch_size, vocab_size).cuda(probs.get_device())
        ones = torch.ones(batch_size, vocab_size).cuda(probs.get_device())
        ref_bow.scatter_add_(-1, targets, ones).detach()
        ref_bow[:,self.pad] = 0
        expected_length = torch.sum(bow).div(batch_size)
        loss = torch.mean(torch.norm(bow-ref_bow,p=1,dim=-1))/ (length_tgt + expected_length)
        return loss

    def compute_ctc_bigram_loss(self, log_probs, logit_mask, targets, blank_index):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)
        targets = targets.tolist()
        probs_blank = probs[:,:,blank_index]
        length = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        length[:,blank_index] = 0
        expected_length = torch.sum(length).div(batch_size)

        logprobs_blank = log_probs[:,:,blank_index]
        cumsum_blank = torch.cumsum(logprobs_blank, dim = -1)
        cumsum_blank_A = cumsum_blank.view(batch_size, 1, length_ctc).expand(-1, length_ctc, -1)
        cumsum_blank_B = cumsum_blank.view(batch_size, length_ctc, 1).expand(-1, -1, length_ctc)
        cumsum_blank_sub = cumsum_blank_A - cumsum_blank_B
        cumsum_blank_sub = torch.cat((torch.zeros(batch_size, length_ctc,1).cuda(cumsum_blank_sub.get_device()), cumsum_blank_sub[:,:,:-1]), dim = -1)
        tri_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([batch_size, length_ctc, length_ctc]).cuda(cumsum_blank_sub.get_device())), 0)
        cumsum_blank_sub = cumsum_blank_sub + tri_mask
        blank_matrix = torch.exp(cumsum_blank_sub)

        gram_1 = []
        gram_2 = []
        gram_count = []
        rep_gram_pos = []
        num_grams = length_tgt - 1
        for i in range(batch_size):
            two_grams = Counter()
            gram_1.append([])
            gram_2.append([])
            gram_count.append([])
            for j in range(num_grams):
                two_grams[(targets[i][j], targets[i][j+1])] += 1
            j = 0
            for two_gram in two_grams:
                if self.pad in two_gram:
                    continue
                gram_1[-1].append(two_gram[0])
                gram_2[-1].append(two_gram[1])
                gram_count[-1].append(two_grams[two_gram])
                if two_gram[0] == two_gram[1]:
                    rep_gram_pos.append((i, j))
                j += 1
            while len(gram_count[-1]) < num_grams:
                gram_1[-1].append(1)
                gram_2[-1].append(1)
                gram_count[-1].append(0)
        gram_1 = torch.LongTensor(gram_1).cuda(blank_matrix.get_device())
        gram_2 = torch.LongTensor(gram_2).cuda(blank_matrix.get_device())
        gram_count = torch.Tensor(gram_count).cuda(blank_matrix.get_device()).view(batch_size, num_grams,1)
        gram_1_probs = torch.gather(probs, -1, gram_1.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, length_ctc, 1)
        gram_2_probs = torch.gather(probs, -1, gram_2.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, 1, length_ctc)
        probs_matrix = torch.matmul(gram_1_probs, gram_2_probs)
        bag_grams = blank_matrix.view(batch_size, 1, length_ctc, length_ctc) * probs_matrix
        bag_grams = torch.sum(bag_grams.view(batch_size, num_grams, -1), dim = -1).view(batch_size, num_grams,1)
        if len(rep_gram_pos) > 0:
            for pos in rep_gram_pos:
                i, j = pos
                gram_id = gram_1[i, j]
                gram_prob = probs[i, :, gram_id]
                rep_gram_prob = torch.sum(gram_prob[1:] * gram_prob[:-1])
                bag_grams[i, j, 0] = bag_grams[i, j, 0] - rep_gram_prob
        match_gram = torch.min(torch.cat([bag_grams,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram).div(batch_size)


        loss = (- 2 * match_gram).div(length_tgt + expected_length - 2)
        
        return loss

    def sequence_ctc_loss_with_logits(self,
                                      logits: torch.FloatTensor,
                                      logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      targets: torch.LongTensor,
                                      target_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      blank_index: torch.LongTensor,
                                      label_smoothing=0,
                                      reduce=True
                                      ) -> torch.FloatTensor:
        # lengths : (batch_size, )
        # calculated by counting number of mask
        logit_lengths = (logit_mask.bool()).long().sum(1)

        if len(targets.size()) == 1:
            targets = targets.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0)
        target_lengths = (target_mask.bool()).long().sum(1)

        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
        log_probs_T = log_probs.transpose(0, 1)
        #     assert (target_lengths == 0).any()
        targets = targets.long()
        targets = targets[target_mask.bool()]
        if reduce:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="mean",
                zero_infinity=True,
            )
        else:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="none",
                zero_infinity=True,
            )
            loss = torch.stack([a / b for a, b in zip(loss, target_lengths)])

        n_invalid_samples = (logit_lengths < target_lengths).long().sum()

        if n_invalid_samples > 0:
            logger.warning(
                f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
            )
            # raise ValueError

        if label_smoothing > 0:
            smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
            loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss
        return loss
    
    

    
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, transcript_tokens, glat, reduce=True, **kwargs
    ):    
        # encoding
        encoder_out = self.encoder(src_tokens, fbk_lengths=src_lengths, **kwargs)
        prev_output_tokens = self.initialize_output_tokens_by_upsampling(encoder_out)
        prev_output_tokens_mask = prev_output_tokens.ne(self.pad)
        output_length = prev_output_tokens_mask.sum(dim=-1)

        transcript_mask = transcript_tokens.ne(self.pad)
        target_mask = tgt_tokens.ne(self.pad)
        target_length = target_mask.sum(dim=-1) 
        # glat_implemented_here
        glat_info = None
        oracle = None
        keep_word_mask = None
        
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                normalized_logits = self.decoder(
                    normalize=True,
                    prev_output_tokens=prev_output_tokens,
                    encoder_out=encoder_out,
                )

                normalized_logits_T = normalized_logits.transpose(0, 1).float() #T * B * C, float for FP16

                best_aligns = best_alignment(normalized_logits_T, tgt_tokens, output_length, target_length, self.tgt_dict.blank_index, zero_infinity=True)
                #pad those positions with <blank>
                padded_best_aligns = torch.tensor([a + [0] * (normalized_logits_T.size(0) - len(a)) for a in best_aligns], device=prev_output_tokens.device, dtype=prev_output_tokens.dtype)
                oracle_pos = (padded_best_aligns // 2).clip(max=tgt_tokens.size(-1)-1)
                oracle = tgt_tokens.gather(-1, oracle_pos)
                oracle = oracle.masked_fill(padded_best_aligns % 2 == 0, self.tgt_dict.blank_index)
                oracle = oracle.masked_fill(~prev_output_tokens_mask, self.pad)
                
                _,pred_tokens = normalized_logits.max(-1)
                same_num = ((pred_tokens == oracle) & prev_output_tokens_mask).sum(dim=-1)
                keep_prob = ((output_length - same_num) / output_length * glat['context_p']).unsqueeze(-1) * prev_output_tokens_mask.float()

                keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()
        
                glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)

                glat_info = {
                    "glat_acc": (same_num.sum() / output_length.sum()).detach(),
                    "glat_context_p": glat['context_p'],
                    "glat_keep": keep_prob.mean().detach(),
                }
                prev_output_tokens = glat_prev_output_tokens                  
                
                                
        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            oracle=oracle,
            keep_word_mask=keep_word_mask,
        )
        
        
        if self.args.use_ngram:
            ctc_loss = self.sequence_ngram_loss_with_logits(word_ins_out, prev_output_tokens_mask, tgt_tokens, self.tgt_dict.blank_index)
        else:
            ctc_loss = self.sequence_ctc_loss_with_logits(
                logits=word_ins_out,
                logit_mask=prev_output_tokens_mask,
                targets=tgt_tokens,
                target_mask=target_mask,
                blank_index=self.tgt_dict.blank_index,
                label_smoothing=self.args.label_smoothing,
                reduce=reduce
            )
            
        
        # T x B x C -> B x T x C
        x = encoder_out["encoder_out"][0].transpose(0, 1)

        transcript_ins_out = self.decoder.output_layer(x)
        encoder_padding_mask = ~ (encoder_out["encoder_padding_mask"][0])
        
        
        ctc_encoder_loss = self.sequence_ctc_loss_with_logits(
            logits=transcript_ins_out,
            logit_mask=encoder_padding_mask,
            targets=transcript_tokens,
            target_mask=transcript_mask,
            blank_index=self.tgt_dict.blank_index,
            label_smoothing=self.args.label_smoothing,
            reduce=reduce
        )
        
        
        
        ret_val = {
            "ctc_loss": {"loss": ctc_loss},
            "ctc_encoder_loss": {"loss": ctc_encoder_loss}
        }
        return ret_val, glat_info

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):        
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        output_lengths = (output_masks.bool()).long().sum(-1) 
        output_logits = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )
 
        _scores, _tokens = output_logits.max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())


        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

            

    
    def forward_streaming_decoder(self, decoder_out, encoder_out, **kwargs):   #TODO     

        output_tokens = decoder_out.output_tokens

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        output_logits = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=0,
        )
   
        _, _tokens = output_logits.max(-1)
        output_tokens = output_tokens.clone()
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])

        return output_tokens[0]
    

    def initialize_output_tokens_with_length(self, src_tokens, length_tgt):
        max_length = length_tgt.max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).long().fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos        
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        return initial_output_tokens
    
    def initialize_output_tokens_by_upsampling(self, encoder_out):
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_tokens = encoder_out["encoder_padding_mask"][0]
        else:
            T, B, _ = encoder_out["encoder_out"][0].shape
            src_tokens = torch.zeros(B, T).bool().to(encoder_out["encoder_out"][0].device)
        src_tokens = src_tokens.long()
        
        src_lengths = src_tokens.ne(self.tgt_dict.pad()).sum(dim=-1).cpu()
        length_unit = torch.ceil(src_lengths / self.unit_size).to(src_tokens.device) #if self.unit_size =2: every 80ms (2 frames)
        length_tgt = (length_unit * self.src_upsample_ratio).long() #src_upsample_ratio be 1 

        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)
        
    
    def initialize_output_tokens(self, encoder_out, src_tokens):
        initial_output_tokens = self.initialize_output_tokens_by_upsampling(encoder_out)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

        

class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, main_context, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.main_context = main_context
        self.src_upsample_ratio = args.src_upsample_ratio
        self.wait_until = args.wait_until
        self.unit_size = args.unit_size
        

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = ModifiedTransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, oracle=None, keep_word_mask=None, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            oracle=oracle,
            keep_word_mask=keep_word_mask,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        oracle=None,
        keep_word_mask=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy:
            #src_embd = encoder_out["encoder_embedding"][0]
            src_embd = encoder_out["encoder_out"][0].detach().transpose(0, 1)
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )
            
            
            num_unit = prev_output_tokens.size(-1) // self.src_upsample_ratio
            pooled_embd, pooled_src_mask = _mean_pooling_in_block(src_embd.transpose(0, 1), src_mask, self.unit_size, num_unit)

            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                self.forward_copying_source(
                    pooled_embd, pooled_src_mask, prev_output_tokens.ne(self.padding_idx)
                ),
            )
            if oracle is not None:
                oracle_embedding, _ = self.forward_embedding(oracle)
                x = x.masked_fill(keep_word_mask.unsqueeze(-1), 0) + oracle_embedding.masked_fill(~keep_word_mask.unsqueeze(-1), 0)
                

        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        dim_t_src = encoder_out["encoder_out"][0].size(0)
        dim_t_block = math.ceil(dim_t_src / self.main_context)
        #dim_t_left = (dim_t_src % self.main_context)
         
        cross_attn_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim_t_block, dim_t_block])), 1 + self.wait_until
            ).to(x)
        cross_attn_mask = torch.repeat_interleave(cross_attn_mask, repeats=self.main_context, dim=1)
        cross_attn_mask = cross_attn_mask[:,:dim_t_src].contiguous()

        cross_attn_mask = torch.repeat_interleave(cross_attn_mask, repeats=(int(self.main_context / self.unit_size * self.src_upsample_ratio)), dim=0)
        cross_attn_mask = cross_attn_mask[:x.size(0),:].contiguous()
        
        block_mask = torch.ones([int(self.main_context / self.unit_size * self.src_upsample_ratio),int(self.main_context / self.unit_size * self.src_upsample_ratio)], dtype=torch.bool, device=x.device)
        block_mask_list = [block_mask for i in range(dim_t_block)]
        block_diag_mask = torch.block_diag(*block_mask_list)
        self_attn_mask = block_diag_mask + torch.tril(torch.ones_like(block_diag_mask), 0)
        self_attn_mask = utils.fill_with_neg_inf(torch.zeros([self_attn_mask.size(0), self_attn_mask.size(0)], device=x.device)).masked_fill(self_attn_mask, 0).to(x)
        self_attn_mask = self_attn_mask[:x.size(0),:x.size(0)]

        
        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=decoder_padding_mask,
                cross_attn_mask=cross_attn_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding


class ModifiedTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    '''
    modify the forward function to add the ''cross_attn_mask'' argument
    '''
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                attn_mask=cross_attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None



@register_model_architecture(
    "nonautoregressive_streaming_speech_transformer_segment_to_segment", "nonautoregressive_streaming_speech_transformer_segment_to_segment"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.src_upsample_ratio = getattr(args, "src_upsample_ratio", 1)
    args.unit_size = getattr(args, "unit_size", 2)
    args.wait_until = getattr(args, "wait_until", 0)
    
    # --- speech arguments ---
    args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 300)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.conv_type= getattr(args, "conv_type", "shallow2d_base")
    args.no_audio_positional_embeddings = getattr(
        args, "no_audio_positional_embeddings", False
    )
    args.main_context = getattr(args, "main_context", 32)
    args.right_context = getattr(args, "right_context", 16)
    args.encoder_max_relative_position = getattr(args, "encoder_max_relative_position", 32)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)