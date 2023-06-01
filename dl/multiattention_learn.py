import torch
import torch.nn as nn

embed_dim = 512
num_heads = 8

query = torch.randn(32,10,embed_dim)
key,value = query,query

multihead_attn = nn.MultiheadAttention(embed_dim,num_heads)
attn_outputs, attn_output_weights = multihead_attn(query,key,value)

print(attn_outputs)
print(attn_output_weights)