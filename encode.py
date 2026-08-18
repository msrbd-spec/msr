#!/usr/bin/env python3
"""
Encode all PNG files in a directory tree into a single JSON archive.

The folder structure is preserved by using relative file paths as keys
in the JSON object. PNG binary data is base64-encoded (lossless).

Usage:
    python encode_pngs.py
    python encode_pngs.py --input ./my_folder --output archive.json
"""

import argparse
import base64
import json
import os
import sys


def encode_pngs(input_dir: str, output_file: str) -> int:
    """Walk input_dir, base64-encode every .png file, write to output_file."""
    input_dir = os.path.abspath(input_dir)
    archive = {}
    count = 0

    for root, dirs, files in os.walk(input_dir):
        for filename in sorted(files):
            if not filename.lower().endswith(".png"):
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, input_dir)
            # Normalize path separators to forward slashes for portability
            rel_path = rel_path.replace(os.sep, "/")

            with open(filepath, "rb") as f:
                binary_data = f.read()

            b64_data = base64.b64encode(binary_data).decode("ascii")
            archive[rel_path] = b64_data
            count += 1
            print(f"  Encoded: {rel_path}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {count} PNG file(s) encoded into '{output_file}'.")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Encode all PNG files into a single JSON archive."
    )
    parser.add_argument(
        "--input", "-i",
        default=".",
        help="Base directory to scan for PNG files (default: current directory).",
    )
    parser.add_argument(
        "--output", "-o",
        default="png_archive.json",
        help="Output JSON file path (default: png_archive.json).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning '{args.input}' for PNG files...")
    encode_pngs(args.input, args.output)


if __name__ == "__main__":
    main()
