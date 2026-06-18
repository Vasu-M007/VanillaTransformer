import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Positional_encodings import PositionalEncodings
from Masked_MHA import MaskedMHA, CrossAttention
from Encoder_block import TransformerEncoder

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        seq_length : int,
        hidden_dim : int,
        num_heads : int,
        dropout_p : float,
        layer_dimension : int,
        num_layers : int,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=8000, embedding_dim=512)
        self.positional_encoding = PositionalEncodings(seq_length,hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_projection = nn.Linear(hidden_dim,8000)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(
            layer_dimension,dropout_p,num_heads,hidden_dim,seq_length) for _ in range(num_layers)])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, token_ids_fr,encoder_hidden_states,src_padding_mask):
        tgt_padding_mask = (token_ids_fr == 0)
        embeddings = self.embed(token_ids_fr) * (self.embed.embedding_dim ** 0.5)
        x = self.positional_encoding.forward(embeddings)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(
            x,encoder_hidden_states,
            tgt_padding_mask=tgt_padding_mask,
            src_padding_mask=src_padding_mask)

        logits = self.linear_projection(x)
        return logits

class DecoderBlock(nn.Module):
    def __init__(self,
        layer_dim : int,
        dropout_p : float, 
        num_heads : int, 
        hidden_dim : int,
        seq_length : int,
        ):

        super().__init__()

        self.mmha = MaskedMHA(num_heads,hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim,num_heads)
        self.ffnn = nn.Sequential(
            nn.Linear(hidden_dim,layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim,hidden_dim)
        )

        self.dropout1 = nn.Dropout(p = dropout_p)
        self.dropout2 = nn.Dropout(p = dropout_p)
        self.dropout3 = nn.Dropout(p = dropout_p)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)


    def forward(self,x, encoder_hidden_states,tgt_padding_mask=None, src_padding_mask=None):
        output = self.dropout1(self.mmha.forward(x,tgt_padding_mask))
        x = self.layer_norm1(x + output)

        output = self.dropout2(self.cross_attn.forward(x,encoder_hidden_states,src_padding_mask))
        x = self.layer_norm2(x + output)

        output = self.dropout3(self.ffnn(x))
        x = self.layer_norm3(x + output)

        return x
    
token_ids_fr = torch.load('train_fr_ids.pt')
token_ids_en = torch.load('train_en_ids.pt')
    
class TestingDecoder:
    def single_batch_test(self):
        with torch.no_grad():
            encoder_hidden_states = TransformerEncoder(seq_length=44,hidden_dim=512,
                                        num_heads=8,dropout_p=0.1,num_layers=6,layer_dimension=2048)
            encoder_hidden_states._reset_parameters()

            input_batch_en = token_ids_en[:32]
            encoder_output = encoder_hidden_states.forward(input_batch_en)

            decoder = TransformerDecoder(49,512,8,0.1,2048,6,encoder_output)
            decoder._reset_parameters()

            input_batch_fr = token_ids_fr[:32]

            decoder_output = decoder.forward(input_batch_fr)

            return decoder_output
        

