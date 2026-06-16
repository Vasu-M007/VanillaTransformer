import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset, random_split
from Transformer import Transformer

device = torch.device("cuda")

token_ids_en = torch.load('train_en_ids.pt')
token_ids_fr = torch.load('train_fr_ids.pt')
token_info_fr = torch.load('token_info_dict_fr.pt')
token_info_en = torch.load('token_info_dict_en.pt')

id_to_token = {v: k for k,v in token_info_fr.items()
}

id_to_token_en = {
    v:k for k,v in token_info_en.items()
}

source = token_ids_en[0].unsqueeze(0).to(device)
source_1 = token_ids_en[0]

for token_id in source_1:
    print(id_to_token_en[token_id.item()])

generated = torch.tensor(
    [[token_info_fr["<SOS>"]]],
    device=device
)

model = Transformer(44,49,512,8,0.1,6,2048).to(device)
checkpoint = torch.load("best_transformer.pth")
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

with torch.no_grad():
    encoder_hidden_states = (model.transformer_encoder(source))
    max_len = 49
    for _ in range(max_len):
        logits = model.transformer_decoder(
            generated,
            encoder_hidden_states
        )

        next_token = (
        logits[:, -1, :].argmax(dim=-1).item()
        )

        generated = torch.cat(
                [
                generated,
                torch.tensor(
                    [[next_token]],
                    device=device
                    )
                ],
                dim=1
        )
        if next_token == token_info_fr["<EOS>"]:
            break

tokens = []

for idx in generated.squeeze(0).tolist():
    tokens.append(
        id_to_token[idx]
    )

print(tokens)




