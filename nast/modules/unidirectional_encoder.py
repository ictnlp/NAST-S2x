
from typing import Dict, Optional
import torch
from torch import Tensor
import torch.nn as nn


from fairseq.modules import TransformerEncoderLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.incremental_decoding_utils import FairseqIncrementalState
from omegaconf import II


from nast.modules.audio_convs import lengths_to_padding_mask
from nast.modules.audio_encoder import AudioTransformerEncoder
from nast.modules.multihead_attention_relative import MultiheadRelativeAttention, replace_relative_attention

import math

class IncrementalEncodingState(FairseqIncrementalState):
    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "buffer")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "buffer", buffer)


def with_incremental_encoding_state(cls):
    cls.__bases__ = (IncrementalEncodingState,) + tuple(
        b for b in cls.__bases__ if b != IncrementalEncodingState
    )
    return cls


@with_incremental_encoding_state
class UnidirectionalConv2D(nn.Module):
    """
        similar to Shallow2D, for online, we remove padding for length, 
        and add prev_extra frame for current block processing
    """
    downsample_ratio = 4
    def __init__(
        self, 
        input_channels,
        input_feat_per_channel,
        conv_out_channels,
        encoder_embed_dim,
    ):
        super().__init__()
        assert input_channels == 1, input_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.kernel_size = 3
        self.pooling_kernel_sizes = [2, 2]
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channels, conv_out_channels, (3,3), stride=(2,1), padding=(0,1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_out_channels,
                conv_out_channels,
                (3,3),
                stride=(2,1),
                padding=(0,1),
            ),
            nn.ReLU(),
        )
        
        conv_agg_dim = input_feat_per_channel * conv_out_channels
        
        self.out_proj = nn.Linear(conv_agg_dim, encoder_embed_dim)
        
        self._extra_frames= 0
        
        for pool_size in self.pooling_kernel_sizes[::-1]:
            self._extra_frames = self._extra_frames * pool_size + self.kernel_size - 1
    
    @property
    def extra_frames(self):
        return self._extra_frames  #should be 6 here

    def forward(
        self, fbank, fbk_lengths,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        bsz, seq_len, _ = fbank.shape
        x = fbank.view(bsz, seq_len, self.input_channels, self.input_feat_per_channel)
        x = x.transpose(1,2).contiguous()

        if incremental_state is not None:
            input_state = self._get_input_buffer(incremental_state)
            curr_x = x
            if "raw_fea" in input_state:
                pre = input_state["raw_fea"]
                pre_length = min(self.extra_frames, pre.shape[2])
                x = torch.cat((pre[:,:,-pre_length:], x), dim=2)
                fbk_lengths += pre_length
                pre = torch.cat((pre, curr_x), dim=2)
                input_state["raw_fea"] = pre
            else:
                input_state["raw_fea"] = curr_x
            incremental_state = self._set_input_buffer(incremental_state, input_state)
        
        x = self.conv(x)

        input_lengths = fbk_lengths - self.extra_frames
        for s in self.pooling_kernel_sizes:
            input_lengths = (input_lengths.float()/s).ceil().long() #exact
        #(B,C,T,fea)->(T,B,C*feature)
        bsz, _, out_seq_len, _ = x.shape
        x = x.permute(2,0,1,3).contiguous().view(out_seq_len, bsz, -1)
        x = self.out_proj(x)
        padding_mask = lengths_to_padding_mask(input_lengths, x)
        
        return x, padding_mask


        
        

@with_incremental_encoding_state
class UnidirectionalTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, main_context=1, right_context=0):
        """
            block length may be infected by dowansmaple, so we get from additional params, not args
        """
        super().__init__(args)
        self.block_size = main_context
        self.right_context = right_context
    
    
    def forward(
        self,
        x,
        encoder_padding_mask,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        rel_pos: Optional[Tensor] = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            #TODO
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
            incremental_state: for online decoding, calculate current 
                `main_context+right_context` frames, and cache `main_context` frames
                for next inference

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e4
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        key = x
        key_padding_mask = encoder_padding_mask
        
        if incremental_state is not None:
            # catch h instead of key, value, maybe waste computation, just for simplify code
            input_state = self.self_attn._get_input_buffer(incremental_state)
            if "prev_key" in input_state and attn_mask is not None:
                #TODO
                prev_len = input_state["prev_key"].shape[2]
                pre_attn_mask = attn_mask.new(attn_mask.shape[0], prev_len).fill_(0)
                attn_mask= torch.cat((pre_attn_mask, attn_mask),dim=1)
                
        if isinstance(self.self_attn, MultiheadRelativeAttention):
            x, _ = self.self_attn(
                query=x,
                key=key,
                value=key,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                incremental_state=incremental_state,
                rel_pos=rel_pos
            )    
        else:        
            x, _ = self.self_attn(
                query=x,
                key=key,
                value=key,
                key_padding_mask=key_padding_mask,
                incremental_state= incremental_state,
                need_weights=False,
                attn_mask=attn_mask,
                
            )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        
        fc_result = x
        
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        
        if self.return_fc and not torch.jit.is_scripting():
            return x, fc_result
        return x

def gen_block_atten_mask(
    x: Tensor,
    padding_mask: Tensor, 
    main_context: int = 1,
    right_context: Tensor = 0
):
    """
    Args:
        x: inpout embedding, TxBxC
    """
    bsz, seq_len = padding_mask.shape
    block_num = seq_len // main_context
    block_idx = torch.arange(seq_len).to(padding_mask.device) // main_context
    pos = torch.arange(seq_len).to(padding_mask.device)
    
    
    if right_context == 0:
        attn_mask = block_idx.unsqueeze(1) < block_idx.unsqueeze(0)
        rel_pos = None
    
    else:   
        with torch.no_grad():
            rc_block_idx = torch.arange(block_num)
            rc_block_pos = rc_block_idx.unsqueeze(1).repeat(1, right_context).view(-1).to(padding_mask.device)
            rc_block_step = (rc_block_idx.unsqueeze(1) + 1) * main_context
            rc_inc_idx = torch.arange(right_context).unsqueeze(0)
            rc_idx = (rc_block_step + rc_inc_idx).view(-1).to(padding_mask.device)
            rc_idx_mask = (rc_idx > (seq_len - 1)).to(padding_mask)
            rc_idx = rc_idx.clamp(0, seq_len - 1)
            
            rc_padding_mask = padding_mask.index_select(1, rc_idx)
            # mask extra length
            rc_padding_mask= rc_padding_mask | rc_idx_mask.unsqueeze(0)
            
            padding_mask = torch.cat((padding_mask, rc_padding_mask), dim=1)
            full_idx = torch.cat((block_idx, rc_block_pos), dim=0)
            attn_mask1 = full_idx.unsqueeze(1) < block_idx.unsqueeze(0)
            attn_mask2 = full_idx.unsqueeze(1).ne(rc_block_pos.unsqueeze(0))
            attn_mask = torch.cat([attn_mask1,attn_mask2], dim=1)
            
            rel_pos = torch.cat((pos,rc_idx))
            
        rc_x = x.index_select(0, rc_idx)
        x = torch.cat((x, rc_x), dim=0)
    
    attn_mask_float = x.new(attn_mask.shape).fill_(0)
    attn_mask_float = attn_mask_float.masked_fill(
        attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
    )

    return x, padding_mask, attn_mask_float, rel_pos
    #return x, padding_mask, attn_mask, rel_pos

@with_incremental_encoding_state
class UnidirectionalAudioTransformerEncoder(AudioTransformerEncoder):
    def __init__(self, args):
        
        ds_ratio = UnidirectionalConv2D.downsample_ratio
        self.ds_ratio = ds_ratio
        mc = args.main_context
        rc = args.right_context
        self.main_context = mc//ds_ratio
        self.right_context= rc//ds_ratio
        assert (self.main_context*ds_ratio == mc) and (self.right_context*ds_ratio == rc)
        super().__init__(args)
        self.extra_frames = self.conv_layers.extra_frames
        
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
    
    @property
    def init_frames(self):
        #return (self.main_context +self.right_context)*self.ds_ratio + self.extra_frames
        return (self.main_context + self.right_context)*self.ds_ratio
    
    @property
    def step_frames(self):
        return self.main_context*self.ds_ratio

    def build_conv_layers(self, args):
        """
            should support incremental_state, extra_frames
        """
        convs= UnidirectionalConv2D(args.input_channels, args.input_feat_per_channel, 64, args.encoder_embed_dim)
        self.extra_frames= convs.extra_frames
        return convs
    
    def build_encoder_layer(self, args):
        layer = UnidirectionalTransformerEncoderLayer(args, self.main_context, self.right_context)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
       
        if getattr(args,"encoder_max_relative_position",-1) > 0:
            layer.self_attn=replace_relative_attention(layer.self_attn, args.encoder_max_relative_position)
        return layer
    
    def forward(
        self,
        fbank: torch.Tensor,
        fbk_lengths: torch.Tensor,
        **kwargs
    ):
        # padding offline

        B,T,C = fbank.shape
        fbk_lengths = fbk_lengths + self.extra_frames
        head = fbank.new(B, self.extra_frames,C).fill_(0)
        fbank = torch.cat((head, fbank), dim=1)

        x, encoder_padding_mask = self.conv_layers(fbank, fbk_lengths, incremental_state=None) # x is already TBC
        curr_frames = x.shape[0]
        
        x = self.embed_scale * x
        fake_tokens = encoder_padding_mask.long()
        # layernorm after garbage convs
        x = self.layernorm_embedding(x)
        if self.embed_positions is not None:
            x = x + self.embed_positions(fake_tokens).transpose(0,1)
            
        attn_mask = None
        
        # build attn_mask for main and right context
        x, encoder_padding_mask, attn_mask, rel_pos = gen_block_atten_mask(
            x, encoder_padding_mask, self.main_context, self.right_context
        )


        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask,
                attn_mask=attn_mask,
                incremental_state=None,
                rel_pos=rel_pos
            )


        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x = x[:curr_frames]
        encoder_padding_mask = encoder_padding_mask[:,:curr_frames]
        
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "dec1_state":[], # reserved for joint decoding
            "dec1_padding_mask":[],
        }

    def forward_infer(
        self,
        fbank:torch.Tensor,
        fbk_lengths:torch.Tensor,
        incremental_state=None,
        finished=False,
        **kwargs
    ):
        
        # padding at the first block (len==0) or offline (None)
        if incremental_state is None or len(incremental_state) == 0:
            B,T,C = fbank.shape
            fbk_lengths += self.extra_frames
            head = fbank.new(B,self.extra_frames,C).fill_(0)
            fbank = torch.cat((head,fbank),dim=1)
        
        x, encoder_padding_mask = self.conv_layers(fbank, fbk_lengths, incremental_state) # x is already TBC
        
        
        fake_tokens = encoder_padding_mask.long()
        # layernorm after garbage convs
        x = self.layernorm_embedding(x)
        if self.embed_positions is not None:
            # cache src_tokens if incremental_state
            if incremental_state is not None:
                input_state = self._get_input_buffer(incremental_state)
                full_tokens = fake_tokens
                if "prev_tokens" in input_state:
                    full_tokens = torch.cat((input_state["prev_tokens"], full_tokens),dim=1)
                pos_emb = self.embed_positions(full_tokens)
                x = x + pos_emb[:, -fake_tokens.shape[1]:].contiguous().transpose(0,1)
                input_state["prev_tokens"] = full_tokens
                incremental_state = self._set_input_buffer(
                    incremental_state, input_state
                )
            else:
                x = x + self.embed_positions(fake_tokens).transpose(0,1)
            
        attn_mask = None
        
        if self.right_context > 0 and incremental_state is not None:
            # cache current input for next block
            input_state = self._get_input_buffer(incremental_state)
            if "rc_input" in input_state:
                pre = input_state["rc_input"].transpose(0,1)
                x = torch.cat([pre, x], dim=0)
                if "rc_mask" in input_state:
                    pre_mask = input_state["rc_mask"]
                else:
                    pre_mask = encoder_padding_mask.new(pre.shape[1], pre.shape[0]).fill_(0)
                encoder_padding_mask = torch.cat((pre_mask, encoder_padding_mask), dim=1)
            rc_input = x[-self.right_context:].transpose(0,1)
            rc_mask = encoder_padding_mask[:, -self.right_context:]
            input_state["rc_input"] = rc_input
            input_state["rc_mask"] = rc_mask
            incremental_state = self._set_input_buffer(
                incremental_state, input_state
            )

        curr_frames = x.shape[0]

        x, encoder_padding_mask, attn_mask, rel_pos = gen_block_atten_mask(
            x, encoder_padding_mask, self.main_context, self.right_context
        )
       

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask,
                attn_mask=attn_mask,
                incremental_state=incremental_state,
                rel_pos=rel_pos
            )


        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        removed_length = x.shape[0]- curr_frames
        
        x = x[:curr_frames]
        
        encoder_padding_mask = encoder_padding_mask[:,:curr_frames]
        
        if not finished and self.right_context >0:
            removed_length += self.right_context
            x = x[:-self.right_context]
            encoder_padding_mask = encoder_padding_mask[:,:-self.right_context]
        
        if incremental_state is not None:
            self.rollback_steps(incremental_state, removed_length)
        
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "dec1_state":[], # reserved for joint decoding
            "dec1_padding_mask":[],
        }

    def rollback_steps(self, incremental_state, removed_length:int):
        if incremental_state is None:
            return
        if removed_length == 0:
            return
        for layer in self.layers:
            input_buffer = layer.self_attn._get_input_buffer(incremental_state)
            input_buffer["prev_key"]= input_buffer["prev_key"][:,:,:-removed_length]
            input_buffer["prev_value"]= input_buffer["prev_value"][:,:,:-removed_length]
            input_buffer["prev_key_padding_mask"] = None #TODO
            layer.self_attn._set_input_buffer(incremental_state, input_buffer)
