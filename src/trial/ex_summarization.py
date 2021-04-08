import torch
import copy
from ft_encoder import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, EncoderBlock, Encoderi

# Abstractive Summarization with Attention Mechanism
print("DUMMY")

print("DUM2")

print("DUM2")

N=6
d_model=512
d_ff=2048
h=8
dropout=0.1
"""Helper: Construct a model from hyperparameters."""
c = copy.deepcopy
attn = MultiHeadedAttention(h, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
position = PositionalEncoding(d_model, dropout)
d = d_model

encoder = EncoderBlock(Encoder(d, c(attn), c(ff), dropout), N)

print(encoder)