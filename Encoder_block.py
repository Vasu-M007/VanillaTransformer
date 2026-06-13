from Positional_encodings import PositionalEncodings
from Multi_head_attention import MultiHeadAttention
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

token_info = torch.load('token_info_dict_en.pt')
token_ids = torch.load('train_en_ids.pt')


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        seq_length : int,
        hidden_dim : int,
        num_heads : int,
        dropout_p : float,
        num_layers : int,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=8000, embedding_dim=512)
        self.positional_encoding = PositionalEncodings(seq_length,hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(num_layers,dropout_p, num_heads,hidden_dim) for _ in range(num_layers)])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, token_ids):
        x = self.embed(token_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x)

        return x

class EncoderBlock(nn.Module):

    def __init__(self, layer_dim : int, dropout_p : float, num_heads : int, hidden_dim : int):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads,hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim,layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, hidden_dim)
        )

        self.dropout1 = nn.Dropout(p = dropout_p)
        self.dropout2 = nn.Dropout(p = dropout_p)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)


    def forward(self, x):
        output = self.dropout1(self.mha.forward(x))
        x = self.layer_norm1(x + output)

        output = self.dropout2(self.feed_forward(x))
        x = self.layer_norm2(x + output)

        return x

class TestingEncoder:
    def single_batch_test(self):
        with torch.no_grad():
            encoder = TransformerEncoder(seq_length=44,hidden_dim=512,
                                        num_heads=8,dropout_p=0.1,num_layers=6)
            encoder._reset_parameters()
            encoder.eval()

            input_batch = token_ids[:32]
            output = encoder.forward(input_batch)

            return output

# y = TestingEncoder()
# print(y.single_batch_test())


