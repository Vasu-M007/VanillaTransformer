import torch
import numpy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

token_info = torch.load('token_info_dict_en.pt')
token_ids_en = torch.load('train_en_ids.pt')
token_ids_fr = torch.load('train_fr_ids.pt')


class PositionalEncodings(nn.Module):
    """Sinosuidal Positional Encodings calculation to be put
    as input to the encoder and decoder stack"""
    def __init__(self, seq_length, d_model):
        super().__init__()

        i = torch.arange(0, d_model//2).unsqueeze(0)
        angle_term = 1 / (10000 ** (2 * i / d_model))

        pos = torch.arange(0, seq_length).unsqueeze(1)

        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(pos * angle_term)
        pe[:, 1::2] = torch.cos(pos * angle_term)

        self.register_buffer("pe", pe) #makes pe trainable and pe would not be a part of model's state dict

    def forward(self, token_embeddings):
        seq_len = token_embeddings.size(1)
        token_embeddings = token_embeddings + self.pe[:seq_len]
        return token_embeddings

class Test_PE_Eng(nn.Module):
    def test_a_batch(self):
        with torch.no_grad():
            raw_embeds = nn.Embedding(8000, 512)
            embeddings = raw_embeds(token_ids_en)
            first_batch = embeddings[:32]
            pos_obj = PositionalEncodings(44,512)
            encoder_input = pos_obj.forward(first_batch)

        return encoder_input
    
class Test_PE_French(nn.Module):
    def test_a_batch(self):
        with torch.no_grad():
            raw_embeds = nn.Embedding(8000, 512)
            embeddings = raw_embeds(token_ids_fr)
            first_batch = embeddings[:32]
            pos_obj = PositionalEncodings(49,512)
            encoder_input = pos_obj.forward(first_batch)

        return encoder_input
    
"""---OPTIONAL TESTING SCRIPT"""
object_en = Test_PE_Eng()
object_fr = Test_PE_French()

final_output_en = object_en.test_a_batch()
final_output_fr = object_fr.test_a_batch()

 

    

    


            







