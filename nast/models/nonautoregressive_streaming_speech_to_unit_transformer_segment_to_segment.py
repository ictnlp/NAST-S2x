# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn.functional as F

from fairseq import utils
from nast.generators.s2u_nat_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import ensemble_decoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq import checkpoint_utils

from nast.models.torch_imputer import best_alignment
import math
import logging
logger = logging.getLogger(__name__)

from pathlib import Path

from .nonautoregressive_streaming_speech_transformer_segment_to_segment import NATransformerModel, NATransformerDecoder


DEFAULT_MAX_TEXT_POSITIONS = 1024
DEFAULT_MAX_AUDIO_POSITIONS = 6000
DEFAULT_MAX_UNIT_POSITIONS = 1200



@register_model("nonautoregressive_streaming_speech_to_unit_transformer_segment_to_segment")
class S2UNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, text_decoder, unit_decoder):
        super().__init__(args, encoder, text_decoder)

        self.hidden_upsample_ratio = args.hidden_upsample_ratio
        self.unit_decoder = unit_decoder
        self.tgt_dict_unit = self.unit_decoder.dictionary
        

    @classmethod #TODO
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        
        args.max_source_positions = DEFAULT_MAX_AUDIO_POSITIONS
        args.max_target_positions = DEFAULT_MAX_TEXT_POSITIONS
        args.max_target_audio_positions = DEFAULT_MAX_UNIT_POSITIONS


        text_decoder_embed_tokens = cls.build_embedding(
            args, task.target_dictionary, args.decoder_embed_dim
        )
        
        unit_decoder_embed_tokens = cls.build_embedding(
            args, task.target_dictionary_unit, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args)
        text_decoder = cls.build_text_decoder(args, task.target_dictionary, text_decoder_embed_tokens, encoder.main_context)
        unit_decoder = cls.build_unit_decoder(args, task.target_dictionary_unit, unit_decoder_embed_tokens, encoder.main_context)
        model = cls(args, encoder, text_decoder, unit_decoder)
        
        return model

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions(), self.unit_decoder.max_positions())
    
    
    @classmethod
    def build_text_decoder(cls, args, tgt_dict, embed_tokens, main_context):
        
        decoder = NATransformerDecoderModified(args, tgt_dict, embed_tokens, main_context)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        pretraining_path = getattr(args, "load_pretrained_text_decoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained text decoder from: {pretraining_path}")
        return decoder
    
    
    @classmethod
    def build_unit_decoder(cls, args, tgt_dict, embed_tokens, main_context):
        args.max_target_positions = args.max_target_audio_positions
        decoder = UnitNATransformerDecoder(args, tgt_dict, embed_tokens, main_context)
        return decoder


    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument(
            "--load-pretrained-text-decoder-from",
            type=str,
            metavar="STR",
            help="model to take text decoder weights from (for initialization)",
        )
    
    
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, transcript_tokens, tgt_units, glat, reduce=True, **kwargs
    ):    
        # encoding
        encoder_out = self.encoder(src_tokens, fbk_lengths=src_lengths, **kwargs)
        prev_output_tokens = self.initialize_output_tokens_by_upsampling(encoder_out)
        prev_output_tokens_mask = prev_output_tokens.ne(self.pad)
        output_length = prev_output_tokens_mask.sum(dim=-1)

        transcript_mask = transcript_tokens.ne(self.pad)
        target_mask = tgt_tokens.ne(self.pad)
        target_length = target_mask.sum(dim=-1)
        
        prev_output_tokens_unit = self.initialize_unit_output_tokens_by_upsampling(prev_output_tokens)
        prev_output_tokens_mask_unit = prev_output_tokens_unit.ne(self.tgt_dict_unit.pad())
        output_length_unit = prev_output_tokens_mask_unit.sum(dim=-1)
        target_mask_unit = tgt_units.ne(self.tgt_dict_unit.pad())
        target_length_unit = target_mask_unit.sum(dim=-1)
        
        #TODO
        assert self.pad == self.tgt_dict_unit.pad()
        
        # glat_implemented_here
        glat_info = None
        oracle = None
        keep_word_mask = None
        
        oracle_unit = None
        keep_word_mask_unit = None
        
        if glat and (tgt_tokens is not None) and (tgt_units is not None):
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                normalized_logits, features = self.decoder(
                    normalize=True,
                    prev_output_tokens=prev_output_tokens,
                    encoder_out=encoder_out,
                )
                
                normalized_logits_unit = self.unit_decoder(
                    normalize=True,
                    prev_output_tokens=prev_output_tokens_unit,
                    text_decoder_out=features,
                    text_decoder_mask=prev_output_tokens_mask,
                    encoder_out=encoder_out,
                )

                normalized_logits_T = normalized_logits.transpose(0, 1).float() #T * B * C, float for FP16
                normalized_logits_T_unit = normalized_logits_unit.transpose(0, 1).float() #T * B * C, float for FP16

                best_aligns = best_alignment(normalized_logits_T, tgt_tokens, output_length, target_length, self.tgt_dict.blank_index, zero_infinity=True)
                best_aligns_unit = best_alignment(normalized_logits_T_unit, tgt_units, output_length_unit, target_length_unit, self.tgt_dict_unit.blank_index, zero_infinity=True)
                #pad those positions with <blank> TODO
                padded_best_aligns = torch.tensor([a + [0] * (normalized_logits_T.size(0) - len(a)) for a in best_aligns], device=prev_output_tokens.device, dtype=prev_output_tokens.dtype)
                padded_best_aligns_unit = torch.tensor([a + [0] * (normalized_logits_T_unit.size(0) - len(a)) for a in best_aligns_unit], device=prev_output_tokens_unit.device, dtype=prev_output_tokens_unit.dtype)
                oracle_pos = (padded_best_aligns // 2).clip(max=tgt_tokens.size(-1)-1)
                oracle_pos_unit = (padded_best_aligns_unit // 2).clip(max=tgt_units.size(-1)-1)
                oracle = tgt_tokens.gather(-1, oracle_pos)
                oracle_unit = tgt_units.gather(-1, oracle_pos_unit)
                oracle = oracle.masked_fill(padded_best_aligns % 2 == 0, self.tgt_dict.blank_index)
                oracle_unit = oracle_unit.masked_fill(padded_best_aligns_unit % 2 == 0, self.tgt_dict_unit.blank_index)
                oracle = oracle.masked_fill(~prev_output_tokens_mask, self.pad)
                oracle_unit = oracle_unit.masked_fill(~prev_output_tokens_mask_unit, self.tgt_dict_unit.pad())
                
                _,pred_tokens = normalized_logits.max(-1)
                _,pred_tokens_unit = normalized_logits_unit.max(-1)
                same_num = ((pred_tokens == oracle) & prev_output_tokens_mask).sum(dim=-1)
                same_num_unit = ((pred_tokens_unit == oracle_unit) & prev_output_tokens_mask_unit).sum(dim=-1)
                keep_prob = ((output_length - same_num) / output_length * glat['context_p']).unsqueeze(-1) * prev_output_tokens_mask.float()
                keep_prob_unit = ((output_length_unit - same_num_unit) / output_length_unit * glat['context_p_unit']).unsqueeze(-1) * prev_output_tokens_mask_unit.float()

                keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()
                keep_word_mask_unit = (torch.rand(prev_output_tokens_unit.shape, device=prev_output_tokens_unit.device) < keep_prob_unit).bool()
        
                glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
                glat_prev_output_tokens_unit = prev_output_tokens_unit.masked_fill(keep_word_mask_unit, 0) + oracle_unit.masked_fill(~keep_word_mask_unit, 0)

                glat_info = {
                    "glat_acc": (same_num.sum() / output_length.sum()).detach(),
                    "glat_acc_unit": (same_num_unit.sum() / output_length_unit.sum()).detach(),
                    "glat_context_p": glat['context_p'],
                    "glat_context_p_unit": glat['context_p_unit'],
                    "glat_keep": keep_prob.mean().detach(),
                    "glat_keep_unit": keep_prob_unit.mean().detach(),
                }
                prev_output_tokens = glat_prev_output_tokens                  
                prev_output_tokens_unit = glat_prev_output_tokens_unit
                                
        # decoding
        word_ins_out, features = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            oracle=oracle,
            keep_word_mask=keep_word_mask,
        )
        
        unit_ins_out = self.unit_decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens_unit,
            encoder_out=encoder_out,
            text_decoder_out=features,
            text_decoder_mask=prev_output_tokens_mask,
            oracle=oracle_unit,
            keep_word_mask=keep_word_mask_unit,
        )
        
        if self.args.use_ngram_unit:
            ctc_unit_loss = self.sequence_ngram_loss_with_logits(unit_ins_out, prev_output_tokens_mask_unit, tgt_units, self.tgt_dict_unit.blank_index)
            
        else:
            if self.args.use_convex_unit:
                ctc_unit_loss = self.convex_sequence_ctc_loss_with_logits(
                    logits=unit_ins_out,
                    logit_mask=prev_output_tokens_mask_unit,
                    targets=tgt_units,
                    target_mask=target_mask_unit,
                    blank_index=self.tgt_dict_unit.blank_index,
                    label_smoothing=self.args.label_smoothing_unit,
                    reduce=reduce
            )
            else:
                ctc_unit_loss = self.sequence_ctc_loss_with_logits(
                    logits=unit_ins_out,
                    logit_mask=prev_output_tokens_mask_unit,
                    targets=tgt_units,
                    target_mask=target_mask_unit,
                    blank_index=self.tgt_dict_unit.blank_index,
                    label_smoothing=self.args.label_smoothing_unit,
                    reduce=reduce
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

        word_ins_out = self.decoder.output_layer(x)
        encoder_padding_mask = ~ (encoder_out["encoder_padding_mask"][0])
        
        
        ctc_encoder_loss = self.sequence_ctc_loss_with_logits(
            logits=word_ins_out,
            logit_mask=encoder_padding_mask,
            targets=transcript_tokens,
            target_mask=transcript_mask,
            blank_index=self.tgt_dict.blank_index,
            label_smoothing=self.args.label_smoothing,
            reduce=reduce
        )
        
        ret_val = {
            "ctc_unit_loss": {"loss": ctc_unit_loss},
            "ctc_loss": {"loss": ctc_loss},
            "ctc_encoder_loss": {"loss": ctc_encoder_loss},
        }
        return ret_val, glat_info
    
    
    def convex_sequence_ctc_loss_with_logits(self,
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
        
        negative_log_losses = F.ctc_loss(
            log_probs_T.float(),  # compatible with fp16
            targets,
            logit_lengths,
            target_lengths,
            blank=blank_index,
            reduction="none",
            zero_infinity=True,
        )
        length_normalized_log_losses = - torch.stack([a / b for a, b in zip(negative_log_losses, target_lengths)])
        
        order = 3
        length_normalized_log_losses = length_normalized_log_losses * order
        losses = torch.exp(length_normalized_log_losses)
        loss = - losses.mean()

        n_invalid_samples = (logit_lengths < target_lengths).long().sum()

        if n_invalid_samples > 0:
            logger.warning(
                f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
            )
            # raise ValueError

        
        return loss


    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):        
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        output_units = decoder_out.output_units
        output_scores_unit = decoder_out.output_scores_unit

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        output_lengths = (output_masks.bool()).long().sum(-1) 
        output_logits, features = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )
        
        # execute the unit decoder
        output_masks_unit = output_units.ne(self.tgt_dict_unit.pad())
        
        output_logits_unit = self.unit_decoder(
            normalize=True,
            prev_output_tokens=output_units,
            encoder_out=encoder_out,
            text_decoder_out=features,
            text_decoder_mask=output_masks,
            step=step,
        )
        
        
 
        _scores, _tokens = output_logits.max(-1)
        _scores_unit, _units = output_logits_unit.max(-1)
        
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        
        output_units.masked_scatter_(output_masks_unit, _units[output_masks_unit])
        output_scores_unit.masked_scatter_(output_masks_unit, _scores_unit[output_masks_unit])
        
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_units=output_units,
            output_scores_unit=output_scores_unit,
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    
    def forward_streaming_decoder(self, decoder_out, encoder_out, **kwargs):   #TODO     

        output_tokens = decoder_out.output_tokens
        output_units = decoder_out.output_units

        # execute the text decoder
        output_masks = output_tokens.ne(self.pad)
        output_logits, features = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=0,
        )
   
        _, _tokens = output_logits.max(-1)
        output_tokens = output_tokens.clone()
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])

        # execute the unit decoder
        output_masks_unit = output_units.ne(self.tgt_dict_unit.pad())
        
        output_logits_unit = self.unit_decoder(
            normalize=True,
            prev_output_tokens=output_units,
            encoder_out=encoder_out,
            text_decoder_out=features,
            text_decoder_mask=output_masks,
            step=0,
        )
        _, _units = output_logits_unit.max(-1)
        output_units = output_units.clone()
        output_units.masked_scatter_(output_masks_unit, _units[output_masks_unit])

        return output_tokens[0], output_units[0]
    

    def initialize_output_tokens(self, encoder_out, src_tokens):
        initial_output_tokens = self.initialize_output_tokens_by_upsampling(encoder_out)
        initial_output_units = self.initialize_unit_output_tokens_by_upsampling(initial_output_tokens)
        
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])
        initial_output_scores_unit = initial_output_units.new_zeros(
            *initial_output_units.size()
        ).type_as(encoder_out["encoder_out"][0])


        return DecoderOut(
            output_units=initial_output_units,
            output_scores_unit=initial_output_scores_unit,
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )
    


    def initialize_unit_output_tokens_by_upsampling(self, prev_output_tokens):
        if self.hidden_upsample_ratio <= 1:
            return prev_output_tokens

        def _us(x, s):
            B = x.size(0)
            _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
            return _x

        return _us(prev_output_tokens, self.hidden_upsample_ratio)

class NATransformerDecoderModified(NATransformerDecoder):

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
        
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out, features # features: B x T x C


class UnitNATransformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, main_context, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, main_context, no_encoder_attn=no_encoder_attn
        )
        self.hidden_upsample_ratio = args.hidden_upsample_ratio
        self.src_upsample_ratio = self.src_upsample_ratio * self.hidden_upsample_ratio
        self.hidden_embedding_copy = True

    @ensemble_decoder
    def forward(self, normalize, encoder_out, text_decoder_out, text_decoder_mask, prev_output_tokens, step=0, oracle=None, keep_word_mask=None, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            text_decoder_out=text_decoder_out,
            text_decoder_mask=text_decoder_mask,
            embedding_copy=(step == 0) & self.hidden_embedding_copy,
            oracle=oracle,
            keep_word_mask=keep_word_mask,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        text_decoder_out=None,
        text_decoder_mask=None,
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
            src_embd = text_decoder_out # B × T × C        
            src_mask = text_decoder_mask # B × T
                     
            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                self.forward_copying_source(
                    src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
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
        #self_attn_mask[:(self.wait_until + 1)*self.src_upsample_ratio,:(self.wait_until + 1)*self.src_upsample_ratio] = 0
        
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











@register_model_architecture(
    "nonautoregressive_streaming_speech_to_unit_transformer_segment_to_segment", "nonautoregressive_streaming_speech_to_unit_transformer_segment_to_segment"
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
    args.hidden_upsample_ratio = getattr(args, "hidden_upsample_ratio", 6)
    
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