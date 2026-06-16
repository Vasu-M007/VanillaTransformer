import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Positional_encodings import PositionalEncodings
from Masked_MHA import MaskedMHA, CrossAttention
from Encoder_block import TransformerEncoder
from Decoder_Block import TransformerDecoder

token_info = torch.load('token_info_dict_en.pt')
token_ids_en = torch.load('train_en_ids.pt')
token_ids_fr = torch.load('train_fr_ids.pt')

class Transformer(nn.Module):
    def __init__(
        self,
        src_seq_length : int,
        tgt_seq_length : int,
        hidden_dim : int,
        num_heads : int,
        dropout_p : float,
        num_layers : int,
        layer_dimension : int,
    ):
        super().__init__()

        self.transformer_encoder = TransformerEncoder(
        src_seq_length,hidden_dim,num_heads,dropout_p,
        num_layers,layer_dimension)

        self.transformer_decoder = TransformerDecoder(
            tgt_seq_length,hidden_dim,num_heads,dropout_p,layer_dimension,num_layers)
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        
        
    def forward(self, token_ids_en, token_ids_fr):
        encoder_hidden_states = self.transformer_encoder.forward(token_ids_en)
        final_output = self.transformer_decoder.forward(token_ids_fr,encoder_hidden_states)

        return final_output
    

class TestTransformer:
    def single_batch_test(self):
        with torch.no_grad():
            output = Transformer(44,49,512,8,0.1,6,2048)

            output.eval()
            output._reset_parameters()

            input_batch_en = token_ids_en[:32]
            input_batch_fr = token_ids_fr[:32]
            decoder_input = input_batch_fr[:, :-1] #right shifted shape:[32,48,512]

            logits = output.forward(input_batch_en, decoder_input)

        return logits



    

        




        