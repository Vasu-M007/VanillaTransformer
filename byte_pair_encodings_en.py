import os
from collections import defaultdict
import torch
import time

words_frequencies = defaultdict(int)

with open('archive(1)/train.en', 'r') as corpus:
    for line in corpus:
        words = line.split()
        for word in words:
            words_frequencies[word] += 1


alphabets = []
for word in words_frequencies.keys():
    for alphabet in word:
        if alphabet not in alphabets:
            alphabets.append(alphabet)
    
alphabets.sort()

letter_splits = {word : [c for c in word] for word in words_frequencies.keys()}

def count_pairs_freqs(letter_splits):
    pair_frequencies = defaultdict(int)
    for word,freq in words_frequencies.items():
        split = letter_splits[word]
        if (len(split) == 1):
            continue

        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_frequencies[pair] += freq
    
    return pair_frequencies
        
pair_counts = count_pairs_freqs(letter_splits)

max_freq = None
best_pair = ""

for pair,frequency in pair_counts.items():
    if max_freq is None or frequency > max_freq:
        best_pair = pair
        max_freq = frequency


special_tokens = ["<PAD>","<SOS>", "<EOS>", "<UNK>"]
vocab = special_tokens + alphabets.copy()
merges = {('i','n') : 'in'}
vocab.append("in")

def merge(a,b, letter_splits):
    for word in words_frequencies.keys():
        split = letter_splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a + b] + split[i+2 :]
            else:
                i += 1

            letter_splits[word] = split
    
    return letter_splits

vocab_size = 8000

while len(vocab) < vocab_size:
    pair_freqs = count_pairs_freqs(letter_splits)
    best_pair = ""
    max_freq = None
    for pair,freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    letter_splits = merge(best_pair[0], best_pair[1], letter_splits)
    merges[best_pair] =  best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

def tokenization(line):
    words = line.split()
    letter_splits = [[letter for letter in word] for word in words]
    for pair, merge in merges.items():
        for idx, split in enumerate(letter_splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i+= 1

            letter_splits[idx] = split

    return sum(letter_splits,[])

token_id_dict = {}
for idx, token in enumerate(vocab):
    token_id_dict[token] = idx

with open('archive(1)/train.en', 'r') as corpus:
    all_token_ids = []
    count = 0
    for line in corpus:
        ids = []
        count += 1

        if count % 250 == 0:
            print(count)

        tokenized_line = tokenization(line)
        tokenized_sentences = ['<SOS>'] + tokenized_line + ['<EOS>']
        for token in tokenized_sentences:
            id_val = token_id_dict[token]
            ids.append(id_val)
        all_token_ids.append(ids)

for sentence_ids in all_token_ids:
    if len(sentence_ids) > 44:
        print("Found oversized sentence!")

    while len(sentence_ids) < 44:
        sentence_ids.append(token_id_dict["<PAD>"])

tensor = torch.tensor(all_token_ids)
torch.save(tensor, "train_en_ids.pt")

torch.save(token_id_dict, 'token_info_dict_en.pt')



