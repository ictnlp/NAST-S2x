from typing import Dict, List
import torch
from torch import Tensor
import torch.nn as nn
from fairseq.models import FairseqEncoder
from fairseq.data import Dictionary

from fairseq.modules import (
    FairseqDropout,
    LayerNorm,Fp32LayerNorm,
    LayerDropModuleList,
    TransformerEncoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from nast.modules.rand_pos import PositionalEmbedding
from nast.modules.audio_convs import get_conv


class AudioTransformerEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(Dictionary())
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop
        embed_dim = args.encoder_embed_dim
        self.padding_idx = self.dictionary.pad()
        self.max_source_positions = args.max_source_positions
        self.conv_layers = self.build_conv_layers(args)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                rand_max = args.rand_pos_encoder,
                learned=args.encoder_learned_pos,
            )
            if not args.no_audio_positional_embeddings
            else None
        )
        self.layernorm_embedding = LayerNorm(embed_dim)
   

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
    
    def build_conv_layers(self, args):
        return get_conv(args.conv_type)(args.input_feat_per_channel, args.encoder_embed_dim)
    
    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def forward(
        self,
        fbank:torch.Tensor,
        fbk_lengths:torch.Tensor,
        **kwargs
    ):
        # x is already TBC
        x, padding_mask = self.conv_layers(fbank, fbk_lengths)
        
        fake_tokens = padding_mask.long()
        # layernorm after garbage convs
        x = self.layernorm_embedding(x)
        if self.embed_positions is not None:
            x = x + self.embed_positions(fake_tokens).transpose(0,1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "dec1_state":[], # reserved for joint decoding
            "dec1_padding_mask":[],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]
        
        if len(encoder_out["dec1_state"]) ==0:
            dec1_states=[]
        else:
            dec1_states = [(encoder_out["dec1_state"][0]).index_select(1, new_order)]
        
        if len(encoder_out["dec1_padding_mask"]) == 0:
            dec1_padding_mask= []
        else:
            dec1_padding_mask = [encoder_out["dec1_padding_mask"][0].index_select(0,new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "dec1_state":dec1_states, # TxBxC
            "dec1_padding_mask":dec1_padding_mask, # BxT
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    