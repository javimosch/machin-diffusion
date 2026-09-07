#!/usr/bin/env python3
"""Extract CLIP vocab.json to a simple tab-separated sidecar.
Format: token<TAB>id (one per line)

Usage: python3 extract_vocab.py vocab.json vocab.json.idx
"""
import json, sys

def main():
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp + ".idx"

    with open(inp) as f:
        vocab = json.load(f)

    with open(out, "w", encoding="utf-8") as f:
        for token, id_ in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\t{id_}\n")

    print(f"Extracted {len(vocab)} vocab entries to {out}")

if __name__ == "__main__":
    main()
