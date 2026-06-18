# Transformer from Scratch — English to French Translation

A vanilla Transformer (Vaswani et al., 2017) built from the ground up in PyTorch — no `nn.Transformer`, no `nn.MultiheadAttention`. Every matrix multiply in the attention mechanism, the encoder/decoder stacks, and the BPE tokenizer is hand-written, mostly to actually understand what's happening inside the black box instead of importing it.

## What's in here

- A byte-pair encoding tokenizer, trained separately for English and French, written from scratch (no `tokenizers` library)
- Sinusoidal positional encodings
- Multi-head self-attention (encoder)
- Masked multi-head self-attention (decoder, causal masking)
- Cross-attention (decoder attending to encoder outputs)
- Full encoder and decoder stacks with residual connections, layer norm, and feed-forward blocks
- A training loop with padding-aware loss and accuracy
- Greedy decoding for inference

## Dataset

English-French sentence pairs from a Kaggle En-Fr dataset, under 50k sentence pairs. Each language gets its own BPE vocabulary, trained independently, capped at 8000 tokens.

## Architecture

| Component | Value |
|---|---|
| Hidden dim | 512 |
| Attention heads | 8 |
| Encoder/decoder layers | 6 each |
| Feed-forward dim | 2048 |
| Vocab size (per language) | 8000 |
| Max source length | 44 |
| Max target length | 49 |
| Dropout | 0.1 |

Standard Transformer shape — nothing exotic. The interesting part isn't the hyperparameters, it's that every layer underneath them is hand-rolled.

## How it's structured

```
byte_pair_encodings_en.py    # trains BPE vocab + tokenizes English corpus
byte_pair_encodings_fr.py    # trains BPE vocab + tokenizes French corpus
Positional_encodings.py      # sinusoidal position embeddings
Multi_head_attention.py      # encoder self-attention
Masked_MHA.py                 # decoder masked self-attention + cross-attention
Encoder_block.py             # encoder block + stack
Decoder_Block.py             # decoder block + stack
Transformer.py                # wires encoder + decoder together
training.py                   # training loop, validation, checkpointing
```

### Tokenization

Both BPE scripts follow the same process: split the corpus into words, start from individual characters, and greedily merge the most frequent adjacent pair until the vocab hits 8000 tokens. Each language's merges are learned independently on its own corpus, so the English and French vocabularies aren't aligned with each other — that's expected, they're not supposed to be.

Special tokens (`<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`) sit at the front of both vocabularies in the same order, which matters: it keeps `<PAD>`'s token id consistent across both languages, so a single `PAD_ID` works for masking on either side of the model without a lookup.

### Attention

The three attention variants share the same core mechanic — split into heads, project to Q/K/V, scaled dot-product, recombine — but each needed something different:

- **Self-attention (encoder)** needs to ignore padded positions in the source sentence, so it never attends to filler tokens.
- **Masked self-attention (decoder)** needs a causal mask on top of the padding mask — a token can't peek at positions ahead of it in the sequence it's generating, and the mask is built fresh for whatever sequence length is passed in, rather than fixed to a training-time length. That second part matters more than it sounds: it's what makes the same module usable both during training (full sequence at once) and during decoding (one token at a time, growing).
- **Cross-attention (decoder → encoder)** lets each decoder position pull in context from the source sentence, masked the same way as encoder self-attention so it ignores source padding.

### Training

Teacher-forced, as is standard: the decoder sees the ground-truth target sequence shifted right by one position, and is trained to predict the next token at every position in parallel. Loss is cross-entropy with padding positions excluded (`ignore_index`). Validation accuracy is computed the same way, masked so padding tokens don't inflate the score.

The best checkpoint (lowest validation loss) is saved after every epoch where it improves.
### Inference

Training doesn't transfer directly to inference — at training time the model sees the correct previous tokens (teacher forcing); at inference it only has what it's generated so far. Greedy decoding starts from `<SOS>`, runs the decoder, takes the highest-probability next token, appends it, and repeats — feeding the model its own output each step until it produces `<EOS>` or hits the max length.

## Running it

```bash
# 1. Build vocabularies and tokenize the corpus
python byte_pair_encodings_en.py
python byte_pair_encodings_fr.py

# 2. Train
python training.py

# 3. Translate (see decode.py / your inference script)
```

Training expects `archive(1)/train.en` and `archive(1)/train.fr` in the working directory, and produces `train_en_ids.pt`, `train_fr_ids.pt`, `token_info_dict_en.pt`, and `token_info_dict_fr.pt` as cached tensors so tokenization doesn't need to be rerun on every training pass.


This is a from-scratch implementation built to understand the Transformer architecture at the level of individual matrix multiplications, not a production-grade translation system. There's no beam search, no KV-caching for faster decoding, no BLEU scoring, and the BPE tokenizer is a straightforward reference implementation rather than an optimized one. All of that is by design — the goal was depth of understanding on the core architecture, not breadth of features.

#Future Scope

- Beam search instead of greedy decoding, for noticeably better translation quality on longer sentences
- BLEU score evaluation, to have a real translation-quality metric rather than relying on next-token accuracy alone
- KV-caching during decoding, to avoid recomputing attention over the entire generated sequence at every step
- A learning rate warmup schedule, closer to the original paper's training recipe
