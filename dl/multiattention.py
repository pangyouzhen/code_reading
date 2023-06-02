import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import linear, softmax
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter


class MultiheadAttention1(Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=False,) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = Parameter(torch.empty(
            (3 * embed_dim, embed_dim)))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)
        self.in_proj_bias = Parameter(
            torch.empty(3 * embed_dim))
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,) -> Tuple[Tensor, Optional[Tensor]]:
        query.dim() == 3

        attn_output, attn_output_weights = multi_head_attention_forward(
            query, key, value, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.out_proj.weight, self.out_proj.bias,
            )
        return attn_output, attn_output_weights


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor]]:
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads

    def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
    ) -> List[Tensor]:
        E = q.size(-1)
        if k is v:
            if q is k:
                proj = linear(q, w, b)
                proj = proj.unflatten(-1, (3, E)).unsqueeze(
                    0).transpose(0, -2).squeeze(-2).contiguous()
                return proj[0], proj[1], proj[2]
    q, k, v = _in_projection_packed(
        query, key, value, in_proj_weight, in_proj_bias)

    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(
        0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    attn_output_weights = attn_output_weights.view(
        bsz, num_heads, tgt_len, src_len)
    attn_output_weights = attn_output_weights.mean(dim=1)
    return attn_output, attn_output_weights


if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8

    query = torch.randn(32, 10, embed_dim)
    key, value = query, query

    multihead_attn = MultiheadAttention1(embed_dim, num_heads)
    attn_outputs, attn_output_weights = multihead_attn(query, key, value)

    print(attn_outputs)
    print(attn_output_weights)
