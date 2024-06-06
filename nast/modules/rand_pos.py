import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from typing import Any, Optional
import math
from fairseq.modules.positional_embedding import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding
)

def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    rand_max:int = 0,
    learned: bool = False,
):
    if rand_max > 0:
        assert learned ==False, "rand_start with learned positional embedding not implemented"
        m= RandStartSinPositionalEmbedding(
            embedding_dim,
            padding_idx,
            rand_max = rand_max,
            init_size = num_embeddings + padding_idx + 1
        )
    elif learned:
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m

class RandStartSinPositionalEmbedding(nn.Module):
    """
        positional embedding starts index from a random number during training, 
        which is more robust for speech encoder compared to starts from 0
    """
    def __init__(self, embedding_dim, padding_idx, rand_max=1, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = RandStartSinPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.rand_max= rand_max
        self.max_positions = int(1e5)
        self.onnx_trace = False
         
    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    
    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)
        
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.training:
            rand_max= min(self.weights.shape[0] - max_pos, self.rand_max)
            bsz = positions.shape[0]
            rand_pos = (torch.rand(bsz)*rand_max).long().to(positions.device)
            positions += rand_pos.unsqueeze(1)

        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )