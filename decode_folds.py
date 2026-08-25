#!/usr/bin/env python3
"""
Decode fold_N.json archives back into PNG files, restoring the folder structure.

Reads all fold_N.json files from a directory.  Each JSON file's name tells
the script which top-level folder to restore into (e.g. fold_1.json -> Fold_1/).
Keys are relative paths (forward-slash separated) and values are base64-
encoded PNG binary data.

The script recreates the original directory tree:
    Fold_N/FL_Run1_Uniform/centralized/...
    Fold_N/FL_Run1_Uniform/fl/...
    Fold_N/FL_Run2_Heterogeneous/centralized/...
    Fold_N/FL_Run2_Heterogeneous/fl/...

Usage:
    python decode_folds.py
    python decode_folds.py --input-dir ./archives --output-dir ./restored
"""

import argparse
import base64
import glob
import json
import os
import re
import sys


def decode_fold(input_file: str, output_dir: str) -> int:
    """Read a single JSON archive, decode base64, write PNG files to output_dir.

    output_dir is the base directory; the top-level folder name (Fold_N) is
    derived from the JSON filename and prepended to each relative path.
    """
    # Derive the top-level folder name from the JSON filename.
    #   fold_1.json  ->  Fold_1
    basename = os.path.basename(input_file)
    match = re.match(r"^fold_(\d+)\.json$", basename, re.IGNORECASE)
    if not match:
        print(f"  Skipping '{basename}': does not match fold_N.json pattern.")
        return 0

    fold_name = f"Fold_{match.group(1)}"

    with open(input_file, "r", encoding="utf-8") as f:
        archive = json.load(f)

    count = 0
    for rel_path, b64_data in archive.items():
        # Convert forward slashes back to OS-specific separator
        rel_path = rel_path.replace("/", os.sep)
        out_path = os.path.join(output_dir, fold_name, rel_path)

        # Create parent directories as needed
        parent_dir = os.path.dirname(out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        binary_data = base64.b64decode(b64_data)
        with open(out_path, "wb") as f:
            f.write(binary_data)

        count += 1
        print(f"  Decoded: {fold_name}/{rel_path}")

    print(f"  -> {count} PNG file(s) decoded from '{input_file}'.")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Decode fold_N.json archives back into PNG files."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=".",
        help="Directory containing fold_N.json files (default: current directory).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Base directory for restored Fold_N trees (default: current directory).",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Find all fold_N.json files (sorted for deterministic order)
    json_files = sorted(
        glob.glob(os.path.join(input_dir, "fold_*.json"))
    )
    # Filter to only those matching the exact fold_N.json pattern
    pattern = re.compile(r"fold_\d+\.json$", re.IGNORECASE)
    json_files = [f for f in json_files if pattern.search(os.path.basename(f))]

    if not json_files:
        print(f"No fold_N.json files found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    total = 0
    for json_file in json_files:
        print(f"\nDecoding '{json_file}' ...")
        total += decode_fold(json_file, output_dir)

    print(f"\nAll done! {total} PNG file(s) decoded from {len(json_files)} archive(s).")
    print(f"Files restored under '{output_dir}'.")


if __name__ == "__main__":
    main()
