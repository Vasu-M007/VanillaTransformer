from Positional_encodings import PositionalEncodings
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

token_info = torch.load('token_info_dict_en.pt')
token_ids = torch.load('train_en_ids.pt')

matrix = torch.randn(8000, 512)
embeddings = matrix[token_ids]

batch_embeddings = embeddings[:32]

positional_encodings = PositionalEncodings(44,512)
final_encodings = positional_encodings(batch_embeddings)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads : int, hidden_dim : int):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
 
        self.multihead_W_query = nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
        self.multihead_W_key = nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
        self.multihead_W_value = nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
        self.w_o = nn.Parameter(torch.empty(self.head_dim * num_heads, self.hidden_dim))
        self.parameter_reset()

    def parameter_reset(self):
        nn.init.xavier_uniform_(self.multihead_W_query)
        nn.init.xavier_uniform_(self.multihead_W_key)
        nn.init.xavier_uniform_(self.multihead_W_value)
        nn.init.xavier_uniform_(self.w_o)

    def forward(self, input_batch):
        contexts = []
        for head in range(self.num_heads):
            Q = input_batch @ self.multihead_W_query[head]
            K = input_batch @ self.multihead_W_key[head]
            V = input_batch @ self.multihead_W_value[head]

            scores = Q.matmul(K.transpose(-1,-2))

            attention = F.softmax(scores / self.head_dim ** 0.5, dim = -1)

            final_context = attention @ V
            contexts.append(final_context)

        multihead_context = torch.cat(contexts, dim = -1)
        output = multihead_context.matmul(self.w_o)
        return output

  