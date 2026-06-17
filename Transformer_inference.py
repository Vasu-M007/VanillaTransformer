import torch
from Transformer import Transformer

device = torch.device("cuda")

token_info = torch.load('token_info_dict_fr.pt')
checkpoint = torch.load('best_transformer_updated.pth')
token_ids_en = torch.load('train_en_ids.pt')

id_to_token_fr = {
    v:k for k,v in token_info.items()
}


model = Transformer(44, 49, 512, 8, 0.1, 6, 2048)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

def greedy_decode(model, src_tokens, token_info, device, max_len=49):
    """
    model       : trained Transformer model, already on `device`
    src_tokens  : 1D tensor of token ids for ONE source sentence, shape [src_seq_len]
                  (already includes <SOS> and <EOS>, already padded the same way training data was)
    token_info  : the token_id_dict (loaded from token_info_dict_fr.pt or _en.pt)
    max_len     : max length to generate before forcing a stop
    """
    model.eval()

    PAD_ID = token_info["<PAD>"]
    SOS_ID = token_info["<SOS>"]
    EOS_ID = token_info["<EOS>"]

    src_batch = src_tokens.unsqueeze(0).to(device) 

    with torch.no_grad():
        src_padding_mask = (src_batch == PAD_ID)
        encoder_hidden_states = model.transformer_encoder.forward(src_batch)

        decoder_input = torch.tensor([[SOS_ID]], device=device)

        for _ in range(max_len - 1):

            logits = model.transformer_decoder.forward(
                decoder_input,
                encoder_hidden_states,
                src_padding_mask
            )

            next_token_logits = logits[:, -1, :]           
            next_token = next_token_logits.argmax(dim=-1)  

            decoder_input = torch.cat(
                [decoder_input, next_token.unsqueeze(0)],
                dim=1
            )

            if next_token.item() == EOS_ID:
                break

    return decoder_input.squeeze(0) 

src_tokens = token_ids_en[0]

output_ids = greedy_decode(model, src_tokens, token_info, device)
print(output_ids.tolist())

for idx in output_ids.tolist():
    print(id_to_token_fr[idx], end=" ")