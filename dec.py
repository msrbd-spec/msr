#!/usr/bin/env python3
"""
Decode a JSON archive back into PNG files, restoring the folder structure.

Reads a JSON file where keys are relative file paths (bundle-root-relative)
and values are base64-encoded PNG binary data, then writes each PNG to its
original path under the output directory.

Because enc.py stores keys relative to the bundle root, decoding BOTH
archives into the bundle root ('.') restores the exact original structure:

    python dec.py --input fold4_archive.json      --output .
    python dec.py --input fold5_fig_archive.json   --output .

Usage:
    python dec.py
    python dec.py --input fold4_archive.json --output .
"""

import argparse
import base64
import json
import os
import sys


def decode_pngs(input_file: str, output_dir: str) -> int:
    """Read JSON archive, decode base64, write PNG files to output_dir."""
    with open(input_file, "r", encoding="utf-8") as f:
        archive = json.load(f)

    count = 0
    for rel_path, b64_data in archive.items():
        # Convert forward slashes back to OS-specific separator
        rel_path = rel_path.replace("/", os.sep)
        out_path = os.path.join(output_dir, rel_path)

        # Create parent directories as needed
        parent_dir = os.path.dirname(out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        binary_data = base64.b64decode(b64_data)
        with open(out_path, "wb") as f:
            f.write(binary_data)

        count += 1
        print(f"  Decoded: {rel_path}")

    print(f"\nDone! {count} PNG file(s) decoded into '{output_dir}'.")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a JSON archive back into PNG files."
    )
    parser.add_argument(
        "--input", "-i",
        default="archive.json",
        help="Input JSON archive file (default: archive.json).",
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory for restored PNG files (default: current directory).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print(f"Decoding '{args.input}' into '{args.output}'...")
    decode_pngs(args.input, args.output)


if __name__ == "__main__":
    main()
