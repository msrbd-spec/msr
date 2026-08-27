#!/usr/bin/env python3
"""
Decode the three ablation category JSON archives back into PNG files,
restoring the exact ablation_training folder structure.

Reads these JSON files from a directory:

    focal_train_dist.json
    focal_uniform.json
    no_swa.json

Each JSON's keys are paths relative to the ablation_training directory
(e.g. "Fold_1/focal_train_dist/cm_abl_focal_train_dist.png"), and values
are base64-encoded PNG binary data.  The script recreates the original
directory tree:

    <output_dir>/Fold_1/focal_train_dist/...
    <output_dir>/Fold_1/focal_uniform/...
    <output_dir>/Fold_1/no_swa/...
    <output_dir>/Fold_2/...
    ...

Usage:
    python decode_ablation.py
    python decode_ablation.py --input-dir ./archives --output-dir ./restored_ablation
"""

import argparse
import base64
import json
import os
import sys


# The three ablation categories (must match encode_ablation.py).
CATEGORIES = ["focal_train_dist", "focal_uniform", "no_swa"]


def decode_archive(input_file: str, output_dir: str) -> int:
    """Read a single JSON archive, decode base64, write PNG files to output_dir.

    Keys are paths relative to the ablation_training root, so they are
    joined directly with output_dir to recreate the full tree.
    """
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

    print(f"  -> {count} PNG file(s) decoded from '{input_file}'.")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Decode the three ablation category JSON archives back into PNG files."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=".",
        help="Directory containing focal_train_dist.json, focal_uniform.json, no_swa.json (default: current directory).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./restored_ablation",
        help="Base directory for the restored ablation_training tree (default: ./restored_ablation).",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    found = 0
    for cat in CATEGORIES:
        json_file = os.path.join(input_dir, f"{cat}.json")
        if not os.path.isfile(json_file):
            print(f"Warning: '{json_file}' not found, skipping.", file=sys.stderr)
            continue

        found += 1
        print(f"\nDecoding '{json_file}' ...")
        total += decode_archive(json_file, output_dir)

    if found == 0:
        print(
            f"Error: None of the expected JSON files ({', '.join(c + '.json' for c in CATEGORIES)}) "
            f"were found in '{input_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nAll done! {total} PNG file(s) decoded from {found} archive(s).")
    print(f"Files restored under '{output_dir}'.")


if __name__ == "__main__":
    main()
