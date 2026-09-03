#!/usr/bin/env python3
"""
Decode JSON archives back into PNG files, restoring the original folder structure.

Reads one or more JSON files (produced by enc.py) where keys are relative
file paths and values are base64-encoded PNG binary data, then writes each
PNG to its original path under the output directory.

Usage:
    python dec.py
    python dec.py --input dp_fold1.json dp_fold3.json dp_fold5.json
    python dec.py --input-dir ./jsons --output ./outputs/dp_sgd
"""

import argparse
import base64
import json
import os
import sys


def decode_json(json_path: str, output_dir: str) -> int:
    """Decode a single JSON archive into PNG files under output_dir."""
    with open(json_path, "r", encoding="utf-8") as f:
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

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Decode JSON archives back into PNG files."
    )
    parser.add_argument(
        "--input", "-i",
        nargs="*",
        default=None,
        help="One or more JSON archive files to decode.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing JSON archives to decode (all *.json files).",
    )
    parser.add_argument(
        "--output", "-o",
        default="./outputs/dp_sgd",
        help="Output directory for restored PNG files (default: ./outputs/dp_sgd).",
    )
    args = parser.parse_args()

    # Collect list of JSON files to process
    json_files = []

    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)
        for fname in sorted(os.listdir(args.input_dir)):
            if fname.lower().endswith(".json"):
                json_files.append(os.path.join(args.input_dir, fname))
    elif args.input:
        for jf in args.input:
            if not os.path.isfile(jf):
                print(f"Error: Input file '{jf}' does not exist.", file=sys.stderr)
                sys.exit(1)
            json_files.append(jf)
    else:
        # Default: look for the 3 known JSON files in current directory
        defaults = ["dp_fold1.json", "dp_fold3.json", "dp_fold5.json"]
        for jf in defaults:
            if os.path.isfile(jf):
                json_files.append(jf)

    if not json_files:
        print("Error: No JSON files found to decode.", file=sys.stderr)
        print("Use --input <file.json> or --input-dir <dir> to specify.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    total = 0
    for jf in json_files:
        print(f"\n{'='*60}")
        print(f"  Decoding: {jf}")
        print(f"{'='*60}")
        count = decode_json(jf, args.output)
        print(f"  -> {count} PNG(s) restored")
        total += count

    print(f"\n{'='*60}")
    print(f"  Done! {total} PNG file(s) decoded from {len(json_files)} JSON archive(s).")
    print(f"  Output directory: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
