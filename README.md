# Vanilla Transformer From Scratch in PyTorch

A complete implementation of the original **Transformer architecture** proposed in *Attention Is All You Need (Vaswani et al., 2017)* built entirely from scratch in PyTorch for English-to-French machine translation.

This project was developed to gain a deeper understanding of modern sequence-to-sequence models by implementing every major Transformer component manually instead of relying on PyTorch's built-in `nn.Transformer` modules.

## Project Highlights

* Implemented the original Encoder-Decoder Transformer architecture from scratch.
* Built custom Byte Pair Encoding (BPE) tokenizers for English and French datasets.
* Implemented Multi-Head Self Attention, Masked Self Attention, and Cross Attention manually.
* Added source and target padding masks to correctly handle padded sequences.
* Implemented sinusoidal positional encodings from the original paper.
* Trained on the Multi30K English-French translation dataset.
* Added teacher forcing, learning rate warmup scheduling, gradient clipping, and validation tracking.
* Implemented autoregressive greedy decoding for inference.

---

## Architecture

The model follows the original Transformer design:

* 6 Encoder Layers
* 6 Decoder Layers
* Hidden Dimension: 512
* Attention Heads: 8
* Feed Forward Dimension: 2048
* Vocabulary Size: 8000
* Dropout: 0.1

### Encoder Block

Each encoder block consists of:

1. Multi-Head Self Attention
2. Residual Connection + Layer Normalization
3. Feed Forward Network (512 → 2048 → 512)
4. Residual Connection + Layer Normalization

### Decoder Block

Each decoder block consists of:

1. Masked Multi-Head Self Attention
2. Encoder-Decoder Cross Attention
3. Feed Forward Network
4. Residual Connections + Layer Normalization

---

## Tokenization

A custom Byte Pair Encoding (BPE) tokenizer was implemented for both English and French corpora.

Special tokens used:

```text
<PAD>
<SOS>
<EOS>
<UNK>
```

Tokenized sequences were converted into fixed-length tensors and used for training.

---

## Attention Implementation

All attention mechanisms were implemented manually using matrix multiplications and learned projection matrices.

### Multi-Head Self Attention

Used within encoder layers to allow each token to attend to every other token in the source sequence.

### Masked Multi-Head Attention

Used inside decoder layers with a causal mask to prevent the model from attending to future tokens during training.

### Cross Attention

Implemented using:

* Queries from decoder hidden states
* Keys from encoder hidden states
* Values from encoder hidden states

This enables the decoder to attend to relevant regions of the source sentence while generating translations.

---

## Training

Training includes:

* Cross Entropy Loss with padding ignored
* Adam Optimizer
* Gradient Clipping
* Teacher Forcing
* Learning Rate Warmup Scheduler
* Validation Loss and Accuracy Tracking
* Best Model Checkpoint Saving

The dataset was split into:

* 90% Training
* 10% Validation

---

## Inference

Inference is performed using greedy decoding.

The decoder starts with the `<SOS>` token and generates one token at a time until `<EOS>` is produced or the maximum sequence length is reached.

```text
<SOS>
  ↓
Predict Next Token
  ↓
Append Token
  ↓
Repeat
  ↓
<EOS>
```

---

## Example Translation

**Input**

```text
Two young White males are outside near many bushes.
```

**Generated Output**

```text
Deux jeunes hommes blancs sont dehors près de buissons.
```

---

## Repository Structure

```text
Transformer.py                 # Encoder-Decoder Transformer wrapper
Encoder_block.py               # Encoder implementation
Decoder_Block.py               # Decoder implementation
Multi_head_attention.py        # Encoder self-attention
Masked_MHA.py                  # Masked attention & cross attention
Positional_encodings.py        # Sinusoidal positional encodings
byte_pair_encodings_en.py      # English BPE tokenizer
byte_pair_encoding_fr.py       # French BPE tokenizer
Training.py                    # Training pipeline
Transformer_inference.py       # Greedy decoding inference
```

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib

---

## Key Learnings

Through this project I gained hands-on experience with:

* Transformer Architectures
* Attention Mechanisms
* Neural Machine Translation
* Sequence-to-Sequence Learning
* Autoregressive Decoding
* Learning Rate Scheduling
* Training Deep Learning Models from First Principles

This implementation intentionally avoids using high-level Transformer abstractions in order to understand and reproduce the core mechanics of the original architecture.
