# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


from fairseq import utils
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules import MultiheadAttention



def replace_relative_attention(mh_attn:MultiheadAttention, max_relative_position = 20):
    """
        build RelAttn based on given MultiheadAttention, 
        modules before works with fairseq.modules.MultiheadAttention,
    """
    rel_attn = MultiheadRelativeAttention(
        mh_attn.embed_dim, mh_attn.num_heads, mh_attn.kdim, mh_attn.vdim,
        mh_attn.dropout_module.p, bias=True,
        add_bias_kv=False, add_zero_attn=False, self_attention=False,
        encoder_decoder_attention=mh_attn.encoder_decoder_attention,
        max_relative_position=max_relative_position
    )
    return rel_attn

@with_incremental_state
class MultiheadRelativeAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, max_relative_position=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        #relative position embedding
        self.max_relative_position = max_relative_position
        num_embeddings = self.max_relative_position * 2 + 1
        self.relative_keys_embedding = self.relative_embedding(num_embeddings, self.head_dim)
        self.relative_values_embedding = self.relative_embedding(num_embeddings, self.head_dim)


        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    # relative attention
    def relative_embedding(self, num_embeddings, embedding_dim):
        m = nn.Embedding(num_embeddings, embedding_dim)
        #nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.xavier_uniform_(m.weight, gain=1.)

        return m
    def generate_relative_positions_matrix_bypos(self, rel_pos, max_relative_position):
        with torch.no_grad():
            length = rel_pos.shape[0]
            range_mat = rel_pos.expand(length,length)
            dist_mat= range_mat - range_mat.t()
            dist_mat = torch.clamp(dist_mat, -max_relative_position,
                                   max_relative_position)
            dist_mat = dist_mat + max_relative_position
        return dist_mat        
    def generate_relative_positions_matrix(self, length, max_relative_position):
        with torch.no_grad():
            range_mat = torch.arange(length).expand(length, length)
            dist_mat= range_mat - range_mat.t()
            dist_mat = torch.clamp(dist_mat, -max_relative_position,
                                   max_relative_position)
            dist_mat = dist_mat + max_relative_position
        return dist_mat

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, rel_pos = None):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # if self.enable_torch_version and not self.onnx_trace and incremental_state is None and not static_kv:
        #     if self.qkv_same_dim:
        #         return F.multi_head_attention_forward(query, key, value,
        #                                               self.embed_dim, self.num_heads,
        #                                               self.in_proj_weight,
        #                                               self.in_proj_bias, self.bias_k, self.bias_v,
        #                                               self.add_zero_attn, self.dropout,
        #                                               self.out_proj.weight, self.out_proj.bias,
        #                                               self.training, key_padding_mask, need_weights,
        #                                               attn_mask)
        #     else:
        #         return F.multi_head_attention_forward(query, key, value,
        #                                               self.embed_dim, self.num_heads,
        #                                               torch.empty([0]),
        #                                               self.in_proj_bias, self.bias_k, self.bias_v,
        #                                               self.add_zero_attn, self.dropout,
        #                                               self.out_proj.weight, self.out_proj.bias,
        #                                               self.training, key_padding_mask, need_weights,
        #                                               attn_mask, use_separate_proj_weight=True,
        #                                               q_proj_weight=self.q_proj_weight,
        #                                               k_proj_weight=self.k_proj_weight,
        #                                               v_proj_weight=self.v_proj_weight)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q =q*self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            
            if key_padding_mask is not None:
                prev_padding_mask = None
                if 'prev_key_padding_mask' in saved_state:
                    prev_padding_mask= saved_state['prev_key_padding_mask']
                prev_len = k.shape[1] - key_padding_mask.shape[1]
                if prev_len>0:
                    if prev_padding_mask is None:
                        prev_padding_mask = key_padding_mask.new(bsz,prev_len ).fill_(0)
                    key_padding_mask = torch.cat([prev_padding_mask, key_padding_mask], dim=1)
                
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        #relative attention key
        if rel_pos is not None:
            prev_len = src_len -rel_pos.shape[0]
            if incremental_state is None or prev_len ==0:
                relative_positions_matrix = self.generate_relative_positions_matrix_bypos(rel_pos, self.max_relative_position)
            else:
              
                pre_rel_pos= torch.arange(prev_len).to(rel_pos)
                rel_pos= torch.cat([pre_rel_pos, rel_pos + prev_len], dim=0)
                relative_positions_matrix = self.generate_relative_positions_matrix_bypos(rel_pos, self.max_relative_position)
                qlen =q.shape[1]
                relative_positions_matrix = relative_positions_matrix[-qlen:]
        else:
            relative_positions_matrix = self.generate_relative_positions_matrix(src_len, self.max_relative_position) # src_len *src_len
        relative_positions_matrix = relative_positions_matrix.type(torch.cuda.LongTensor)
        #relative_positions_matrix = Variable(relative_positions_matrix, requires_grad=False)
        relations_keys = self.relative_keys_embedding(relative_positions_matrix)[-tgt_len:]
        q_t = q.permute(1,0,2)
        r_t = relations_keys.transpose(1, 2)
        relations_keys_logits = torch.bmm(q_t, r_t)
        #assert list(relations_keys_logits.size()) == [bsz * self.num_heads, tgt_len, src_len]
        attn_weights += relations_keys_logits.transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        #relative attention value
        relations_values = self.relative_values_embedding(relative_positions_matrix)[-tgt_len:]
        attn_weights_t = attn_weights.permute(1,0,2)
        relations_values_attn = torch.bmm(attn_weights_t.float(), relations_values.float()).type_as(attn_weights)
        #assert list(relations_values_attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn += relations_values_attn.transpose(0, 1)

        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights


    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights