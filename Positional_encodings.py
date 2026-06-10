import torch
import numpy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

token_info = torch.load('token_info_dict_en.pt')
token_ids = torch.load('train_en_ids.pt')

matrix = torch.randn(8000, 512)
embeddings = matrix[token_ids]

class PositionalEncodings(nn.Module):

    def __init__(self, seq_length, d_model):
        super().__init__()

        i = torch.arange(0, d_model//2).unsqueeze(0)
        angle_term = 1 / (10000 ** (2 * i / d_model))

        pos = torch.arange(0, seq_length).unsqueeze(1)

        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(pos * angle_term)
        pe[:, 1::2] = torch.cos(pos * angle_term)

        self.register_buffer("pe", pe)

    def forward(self, token_embeddings):
        token_embeddings = token_embeddings + self.pe
        return token_embeddings

positional_encodes = PositionalEncodings(44, 512)
X = positional_encodes(embeddings)
print(X.shape)









