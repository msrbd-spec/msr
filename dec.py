#!/usr/bin/env python3
"""
Decode per-fold JSON archives back into PNG files, restoring the folder structure.

Reads JSON files (e.g. Fold_1.json, Fold_2.json, Fold_3.json) where keys are
relative file paths and values are base64-encoded PNG binary data, then writes
each PNG to its original path under the output directory.

The fold name is derived from the JSON filename (e.g. Fold_1.json -> Fold_1/)
so that the exact original structure is restored:

    output_dir/Fold_1/fedavg_mu0_sev0/cm_fedavg_mu0_sev0.png
    output_dir/Fold_2/fedprox_sev20/roc_fedprox_sev20.png
    ...

Usage:
    python dec.py
    python dec.py --input Fold_1.json,Fold_2.json,Fold_3.json --output ./restored
"""

import argparse
import base64
import json
import os
import sys


def decode_archive(input_file: str, output_dir: str) -> int:
    """Read a single JSON archive, decode base64, write PNG files to output_dir.

    The fold name is derived from the JSON filename (without extension) and
    prepended to every relative path so the full Fold_N/... structure is
    restored.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        archive = json.load(f)

    # Derive fold name from filename, e.g. "Fold_1.json" -> "Fold_1"
    fold_name = os.path.splitext(os.path.basename(input_file))[0]

    count = 0
    for rel_path, b64_data in archive.items():
        # Convert forward slashes back to OS-specific separator
        rel_path = rel_path.replace("/", os.sep)
        # Prepend the fold name to restore the full structure
        out_path = os.path.join(output_dir, fold_name, rel_path)

        # Create parent directories as needed
        parent_dir = os.path.dirname(out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        binary_data = base64.b64decode(b64_data)
        with open(out_path, "wb") as f:
            f.write(binary_data)

        count += 1
        print(f"  Decoded: {os.path.join(fold_name, rel_path)}")

    print(f"\nDone! {count} PNG file(s) decoded from '{input_file}' into '{output_dir}'.")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Decode per-fold JSON archives back into PNG files."
    )
    parser.add_argument(
        "--input", "-i",
        default="Fold_1.json,Fold_2.json,Fold_3.json",
        help="Comma-separated list of JSON archive files (default: Fold_1.json,Fold_2.json,Fold_3.json).",
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory for restored PNG files (default: current directory).",
    )
    args = parser.parse_args()

    input_files = [f.strip() for f in args.input.split(",") if f.strip()]

    os.makedirs(args.output, exist_ok=True)

    total = 0
    for input_file in input_files:
        if not os.path.isfile(input_file):
            print(f"Warning: Input file '{input_file}' does not exist — skipping.",
                  file=sys.stderr)
            continue

        print(f"Decoding '{input_file}' into '{args.output}'...")
        total += decode_archive(input_file, args.output)

    print(f"\nAll done! {total} PNG file(s) decoded from {len(input_files)} archive(s).")


if __name__ == "__main__":
    main()
