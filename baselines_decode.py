#!/usr/bin/env python3
"""
Decode JSON archives back into PNG files, restoring the folder structure.

Reads all *_archive.json files in the input directory. Each JSON file's name
encodes the model name and run label:
  e.g. ConvNeXtV2-Tiny_Run1_archive.json -> model=ConvNeXtV2-Tiny, run=Run1

Keys are relative file paths within the model folder and values are
base64-encoded PNG binary data. The full folder structure is restored:
  <output_dir>/<model_name>/Fold_1/FL_Run1_Uniform/...png

Usage:
    python baselines_decode.py
    python baselines_decode.py --input . --output ./restored
"""

import argparse
import base64
import glob
import json
import os
import sys


def decode_archive(input_file: str, output_dir: str) -> int:
    """Read a single JSON archive, decode base64, write PNG files to output_dir.

    The model name is derived from the archive filename by stripping
    '_Run1_archive.json' or '_Run2_archive.json' (or just '_archive.json').
    Files are restored under <output_dir>/<model_name>/<rel_path>.
    """
    # Derive model name from filename
    basename = os.path.basename(input_file)
    if basename.endswith("_archive.json"):
        # Strip "_archive.json" -> e.g. "ConvNeXtV2-Tiny_Run1"
        stem = basename[: -len("_archive.json")]
        # Strip "_Run1" or "_Run2" suffix if present
        for run_suffix in ("_Run1", "_Run2"):
            if stem.endswith(run_suffix):
                model_name = stem[: -len(run_suffix)]
                break
        else:
            model_name = stem
    else:
        # Fallback: strip .json
        model_name = basename[: -len(".json")]

    with open(input_file, "r", encoding="utf-8") as f:
        archive = json.load(f)

    count = 0
    for rel_path, b64_data in archive.items():
        # Convert forward slashes back to OS-specific separator
        rel_path = rel_path.replace("/", os.sep)
        out_path = os.path.join(output_dir, model_name, rel_path)

        # Create parent directories as needed
        parent_dir = os.path.dirname(out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        binary_data = base64.b64decode(b64_data)
        with open(out_path, "wb") as f:
            f.write(binary_data)

        count += 1
        print(f"  Decoded: {model_name}{os.sep}{rel_path}")

    print(f"  -> {count} PNG file(s) decoded from '{input_file}'.\n")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Decode JSON archives back into PNG files, restoring the folder structure."
    )
    parser.add_argument(
        "--input", "-i",
        default=".",
        help="Directory containing *_archive.json files (default: current directory).",
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory for restored PNG files (default: current directory).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Find all *_archive.json files in the input directory
    archive_files = sorted(glob.glob(os.path.join(args.input, "*_archive.json")))

    if not archive_files:
        print(f"Error: No *_archive.json files found in '{args.input}'.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    total_count = 0
    for archive_file in archive_files:
        print(f"Decoding '{archive_file}'...")
        total_count += decode_archive(archive_file, args.output)

    print(f"Done! {total_count} PNG file(s) total decoded into '{args.output}'.")


if __name__ == "__main__":
    main()
