import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset, random_split
from Transformer import Transformer

token_ids_en = torch.load('train_en_ids.pt')
token_ids_fr = torch.load('train_fr_ids.pt')
token_info = torch.load('token_info_dict_fr.pt')

device = torch.device("cuda")

class TranslationData(Dataset):
    def __init__(self, token_ids_en, token_ids_fr):
        self.token_ids_en = token_ids_en
        self.token_ids_fr = token_ids_fr

    def __len__(self):
        return len(self.token_ids_en)
    
    def __getitem__(self, idx):
        return (self.token_ids_en[idx], self.token_ids_fr[idx])
    
dataset = TranslationData(token_ids_en,token_ids_fr)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

model = Transformer(44,49,512,8,0.1,6,2048)
model._reset_parameters()
model = model.to(device)

PAD_ID = token_info["<PAD>"]

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

warmup_steps = 4000
step_num = 0

def get_lr(step_num, hidden_dim=512, warmup_steps=4000):
    step_num = max(step_num, 1)
    return (hidden_dim ** -0.5) * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)

best_val_loss = float("inf")
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for en_batch, fr_batch in train_loader:

        en_batch = en_batch.to(device)
        fr_batch = fr_batch.to(device)

        decoder_input = fr_batch[:, :-1]
        target = fr_batch[:, 1:]

        logits = model(
            en_batch,
            decoder_input
        )

        loss = criterion(
            logits.reshape(-1,8000),
            target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )   
        step_num += 1
        lr = get_lr(step_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for en_batch, fr_batch in val_loader:

            en_batch = en_batch.to(device)
            fr_batch = fr_batch.to(device)

            decoder_input = fr_batch[:, :-1]
            target = fr_batch[:, 1:]

            logits = model(
                en_batch,
                decoder_input
            )

            loss = criterion(
                logits.reshape(-1, 8000),
                target.reshape(-1)
            )

            val_loss += loss.item()

            preds = logits.argmax(dim=-1)

            mask = target != PAD_ID

            correct += ((preds == target) & mask).sum().item()
            total += mask.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total



    print(
    f"Epoch {epoch+1:02d} | "
    f"Train Loss: {avg_train_loss:.4f} | "
    f"Val Loss: {avg_val_loss:.4f} | "
    f"Val Acc: {val_accuracy:.4f}"
)

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
            },   
            "best_transformer_updated.pth"
        )

        print(
            f"Saved new best model "
            f"(val loss = {avg_val_loss:.4f})"
        )


    

        
