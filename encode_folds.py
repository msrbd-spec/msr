#!/usr/bin/env python3
"""
Encode all PNG files in each Fold_N directory into separate JSON archives.

For every Fold_N folder (N = 1..5), this script walks the entire subtree
(FL_Run1_Uniform/centralized, FL_Run1_Uniform/fl,
 FL_Run2_Heterogeneous/centralized, FL_Run2_Heterogeneous/fl, etc.),
base64-encodes every .png file, and writes a single JSON archive named
fold_N.json.

The folder structure is preserved by using relative file paths (relative
to the Fold_N directory) as keys in the JSON object.  Path separators are
normalized to forward slashes for portability.

Usage:
    python encode_folds.py
    python encode_folds.py --input ./best_architecture --output-dir ./archives
"""

import argparse
import base64
import json
import os
import re
import sys


def encode_fold(fold_dir: str, output_file: str) -> int:
    """Walk fold_dir, base64-encode every .png file, write to output_file.

    Keys in the JSON are paths relative to fold_dir.
    """
    fold_dir = os.path.abspath(fold_dir)
    archive = {}
    count = 0

    for root, dirs, files in os.walk(fold_dir):
        for filename in sorted(files):
            if not filename.lower().endswith(".png"):
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, fold_dir)
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

    print(f"  -> {count} PNG file(s) encoded into '{output_file}'.")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Encode PNG files in each Fold_N into separate JSON archives."
    )
    parser.add_argument(
        "--input", "-i",
        default=".",
        help="Base directory containing Fold_1 .. Fold_5 (default: current directory).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory where fold_N.json files are written (default: current directory).",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Find all Fold_N directories
    fold_pattern = re.compile(r"^Fold_(\d+)$", re.IGNORECASE)
    fold_dirs = []
    for entry in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, entry)
        if os.path.isdir(full_path):
            match = fold_pattern.match(entry)
            if match:
                fold_dirs.append((int(match.group(1)), entry, full_path))

    if not fold_dirs:
        print(f"No Fold_N directories found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    total = 0
    for fold_num, fold_name, fold_path in sorted(fold_dirs):
        print(f"\nProcessing {fold_name} ...")
        output_file = os.path.join(output_dir, f"fold_{fold_num}.json")
        total += encode_fold(fold_path, output_file)

    print(f"\nAll done! {total} PNG file(s) encoded across {len(fold_dirs)} fold(s).")
    print(f"JSON archives written to '{output_dir}'.")


if __name__ == "__main__":
    main()
