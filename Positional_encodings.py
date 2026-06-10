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

# d_q, d_k, d_v = 512,512,512

# W_query = torch.nn.Parameter(torch.randn(d_q,X.size(2))) * 0.02
# W_key = torch.nn.Parameter(torch.randn(d_k,X.size(2))) * 0.02
# W_value = torch.nn.Parameter(torch.randn(d_v,X.size(2))) * 0.02

# query_vectors = X @ W_query
# key_vectors = X @ W_key
# value_vectors = X @ W_value


# similarity_matrix = query_vectors.matmul(key_vectors.transpose(-1,-2))

# attention_weights = F.softmax(similarity_matrix / d_k ** 0.5, dim=-1)

# context_vectors = attention_weights.matmul(value_vectors)

# # attn = attention_weights[0].detach().numpy()

# # plt.imshow(attn)
# # plt.colorbar()
# # plt.xlabel("Key Position")
# # plt.ylabel("Query Position")
# # plt.title("Attention Weights")
# # plt.show()

# num_heads = 8
# head_dim = 64

# multihead_W_query = torch.nn.Parameter(torch.randn(num_heads,512,head_dim)) * 0.02
# multihead_W_key = torch.nn.Parameter(torch.randn(num_heads,512,head_dim)) * 0.02
# multihead_W_value = torch.nn.Parameter(torch.randn(num_heads,512,head_dim)) * 0.02

# batch_X = X[:32]

# contexts = []
# for head in range(num_heads):
#     Q = batch_X @ multihead_W_query[head]
#     K = batch_X @ multihead_W_key[head]
#     V = batch_X @ multihead_W_value[head]

#     scores = Q.matmul(K.transpose(-1,-2))

#     attention = F.softmax(scores / 64 * 0.5, dim = -1)

#     final_context = attention @ V
#     contexts.append(final_context)

# multihead_context = torch.cat(contexts, dim = -1)
# print(multihead_context.shape)











