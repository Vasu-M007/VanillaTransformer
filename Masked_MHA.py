from Positional_encodings import PositionalEncodings
from Encoder_block import  TransformerEncoder
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

token_info = torch.load('token_info_dict_fr.pt')
token_ids_fr = torch.load('train_fr_ids.pt')
token_ids_en = torch.load('train_en_ids.pt')

class MaskedMHA(nn.Module):
    def __init__(self, num_heads : int, hidden_dim : int, seq_length : int):
        """Here Seq length becomes 48 as decoder input is right shifted by 1"""
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.multihead_W_query = nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
        self.multihead_W_key = nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
        self.multihead_W_value = nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
        self.w_o = nn.Parameter(torch.empty(self.head_dim * num_heads, self.hidden_dim))
        self.mask_matrix = torch.triu(torch.ones(seq_length,seq_length),diagonal=1).bool()
        self.register_buffer(
            "masked_matrix",
            torch.triu(torch.ones(seq_length, seq_length),diagonal=1).bool()
        )
        self.parameter_reset()
        
    def parameter_reset(self):
        nn.init.xavier_uniform_(self.multihead_W_query)
        nn.init.xavier_uniform_(self.multihead_W_key)
        nn.init.xavier_uniform_(self.multihead_W_value)
        nn.init.xavier_uniform_(self.w_o)

    def forward(self, decoder_in_batch):
        contexts = []
        for head in range(self.num_heads):
            Q = decoder_in_batch @ self.multihead_W_query[head]
            K = decoder_in_batch @ self.multihead_W_key[head]
            V = decoder_in_batch @ self.multihead_W_value[head]

            scores = Q.matmul(K.transpose(-1,-2))
            scores = scores.masked_fill(self.mask_matrix,float("-inf"))

            attention = F.softmax(scores / self.head_dim ** 0.5, dim = -1)

            final_context = attention @ V
            contexts.append(final_context)

        multihead_context = torch.cat(contexts, dim = -1)
        output = multihead_context.matmul(self.w_o)
        return output


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim : int,
        num_heads : int,
        ):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.encoder_W_key = nn.Parameter(
            torch.empty(num_heads, hidden_dim, self.head_dim))
        
        self.encoder_W_value =nn.Parameter(
            torch.empty(num_heads, hidden_dim, self.head_dim))
        
        self.decoder_W_query = nn.Parameter(
            torch.empty(num_heads, hidden_dim, self.head_dim))
        
        self.w_o = nn.Parameter(torch.empty(self.head_dim * num_heads, self.hidden_dim))
        self.parameter_reset()
        
    def parameter_reset(self):
        nn.init.xavier_uniform_(self.encoder_W_key)
        nn.init.xavier_uniform_(self.encoder_W_value)
        nn.init.xavier_uniform_(self.decoder_W_query)
        nn.init.xavier_uniform_(self.w_o)

    def forward(self, decoder_in_batch, encoder_hidden_states):
        contexts = []
        for head in range(self.num_heads):
            Q = decoder_in_batch @ self.decoder_W_query[head]
            K = encoder_hidden_states.matmul(self.encoder_W_key[head])
            V = encoder_hidden_states.matmul(self.encoder_W_value[head])

            scores = Q.matmul(K.transpose(-1,-2))
    
            attention = F.softmax(scores / self.head_dim ** 0.5, dim = -1)

            final_context = attention @ V
            contexts.append(final_context)

        multihead_context = torch.cat(contexts, dim = -1)
        output = multihead_context.matmul(self.w_o)
        return output
    

    
class TestCrossAttention(nn.Module):
        def single_batch_test(self):
            with torch.no_grad():
                encoder_object = TransformerEncoder(44,512,4,0.1,2)
                input_batch = token_ids_en[:32]
                encoder_out = encoder_object.forward(input_batch)

                masked_multi = MaskedMHA(8,512,48)
                final_out = CrossAttention(512,48,8)
                decoder_input = position_encodes[:, :-1]
                mmha_out = masked_multi.forward(decoder_input)

                out = final_out.forward(mmha_out,encoder_out)

            return out
        
embed = nn.Embedding(num_embeddings=8000, embedding_dim=512)
embeddings = embed(token_ids_fr)
embeddings = embeddings[:32]
encodings = PositionalEncodings(49,512)
position_encodes = encodings.forward(embeddings)
        



        
            











        
