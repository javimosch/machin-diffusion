#!/usr/bin/env python3
"""Tokenize a text prompt into CLIP token IDs for SD-Turbo.
Outputs one integer per line (BOS + tokens + EOS, padded to 77).

Usage: python3 tokenize.py <vocab.json> <merges.txt> "<prompt>" [output_file]
"""
import json, sys, re

def bytes_to_unicode():
    """CLIP's byte-to-unicode mapping."""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("\xa1"), ord("\xac")+1)) + list(range(ord("\xae"), ord("\xff")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev = word[0]
    for char in word[1:]:
        pairs.add((prev, char))
        prev = char
    return pairs

def main():
    vocab_path = sys.argv[1]
    merges_path = sys.argv[2]
    text = sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else "token_ids.txt"

    with open(vocab_path, encoding="utf-8") as f:
        encoder = json.load(f)

    with open(merges_path, encoding="utf-8") as f:
        merges = f.read().split("\n")
    # Skip header line and empty lines
    merges = [m for m in merges if m and not m.startswith("#")]
    bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}

    byte_encoder = bytes_to_unicode()

    # CLIP pattern (simplified — no \p{L} unicode properties)
    pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[^\s\w]+", re.UNICODE)

    bos = 49406
    eos = 49407
    max_len = 77

    tokens = [bos]

    text = text.lower()
    for token in re.findall(pat, text):
        # Encode each token through byte-level BPE
        token = " ".join(byte_encoder[b] for b in token.encode("utf-8"))
        word = tuple(token.split())
        pairs = get_pairs(word)
        if not pairs:
            continue

        while True:
            bigram = min(pairs, key=lambda p: bpe_ranks.get(p, float("inf")))
            if bigram not in bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        for w in word:
            if w in encoder:
                tokens.append(encoder[w])

    tokens.append(eos)
    # Pad to max_len with pad_token_id (0 for OpenCLIP/SD 2.x, NOT eos)
    pad_id = 0
    while len(tokens) < max_len:
        tokens.append(pad_id)
    tokens = tokens[:max_len]

    with open(out_path, "w") as f:
        f.write("\n".join(str(t) for t in tokens) + "\n")

    print(f"Tokenized to {len(tokens)} tokens -> {out_path}")

if __name__ == "__main__":
    main()
