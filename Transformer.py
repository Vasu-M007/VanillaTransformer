import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Positional_encodings import PositionalEncodings
from Masked_MHA import MaskedMHA, CrossAttention
from Encoder_block import TransformerEncoder
from Decoder_Block import TransformerDecoder

class Transformer(nn.Module):
    def __init__(
        self,
        seq_length : int,
        hidden_dim : int,
        num_heads : int,
        dropout_p : float,
        num_layers : int,
        layer_dimension : int,
        encoder_hidden_states : int
    ):
        super().__init__()

        self.transformer_encoder = TransformerEncoder(
        seq_length,hidden_dim,num_heads,dropout_p,
        num_layers,layer_dimension)

        self.transformer_decoder = TransformerDecoder(
            seq_length,hidden_dim,num_heads,dropout_p,layer_dimension,num_layers,encoder_hidden_states)
        
        self.encoder_input = PositionalEncodings(seq_length,hidden_dim)
        self.decoder_input = PositionalEncodings(seq_length,hidden_dim)

    def create_token_embeddings(self, token_ids_en, token_ids_fr):
        embedding = nn.Embedding(8000,512)
        token_embed_en = embedding(token_ids_en)
        token_embed_fr = embedding(token_ids_fr)
        
    def forward(self, token_ids_en, token_ids_fr):
        embedding = nn.Embedding(8000,512)
        token_embeddings = embedding(token_ids_en)

        encoder_hidden_states = self.transformer_encoder.forward(token_ids_en)
        decoder_output = self.tr



        