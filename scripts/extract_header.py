#!/usr/bin/env python3
"""Extract safetensors header to a simple text sidecar (one-time, ~20KB).
Does NOT modify the model — the .safetensors is used as-is via mmap.

Output format (one tensor per line, tab-separated):
    tensor_name<TAB>data_start<TAB>data_end<TAB>ndims<TAB>dim0,dim1,...

Usage: python3 extract_header.py model.safetensors [model.safetensors.idx]
"""
import struct, json, sys

def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path + ".idx"

    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))

    lines = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        shape = info["shape"]
        dims = ",".join(str(d) for d in shape)
        lines.append(f"{name}\t{start}\t{end}\t{len(shape)}\t{dims}")

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Extracted {len(lines)} tensors to {out}")

if __name__ == "__main__":
    main()
