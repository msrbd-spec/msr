#!/usr/bin/env python3
"""
Encode all PNG files in the ablation_training directory tree into three
JSON archives — one per ablation category.

The ablation_training folder has the structure:

    ablation_training/
    ├── Fold_1/
    │   ├── focal_train_dist/   (*.png)
    │   ├── focal_uniform/       (*.png)
    │   └── no_swa/              (*.png)
    ├── Fold_2/  ...
    └── Fold_5/  ...

This script walks the entire tree, base64-encodes every .png file, and
groups them by category into three JSON files:

    focal_train_dist.json
    focal_uniform.json
    no_swa.json

Keys in each JSON are paths relative to the ablation_training directory
(e.g. "Fold_1/focal_train_dist/cm_abl_focal_train_dist.png"), with
forward-slash separators for portability.  Only .png files are encoded;
metrics/history .json files are ignored.

Usage:
    python encode_ablation.py
    python encode_ablation.py --input ./ablation_training --output-dir ./archives
"""

import argparse
import base64
import json
import os
import sys


# The three ablation categories.  A PNG belongs to a category if the
# category name appears as a path component in its relative path.
CATEGORIES = ["focal_train_dist", "focal_uniform", "no_swa"]


def encode_ablation(input_dir: str, output_dir: str) -> int:
    """Walk input_dir, base64-encode every .png, group by category, write 3 JSONs.

    Returns the total number of PNG files encoded.
    """
    input_dir = os.path.abspath(input_dir)

    # One archive dict per category
    archives = {cat: {} for cat in CATEGORIES}
    total = 0

    for root, dirs, files in os.walk(input_dir):
        for filename in sorted(files):
            if not filename.lower().endswith(".png"):
                continue

            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, input_dir)
            # Normalize path separators to forward slashes for portability
            rel_path = rel_path.replace(os.sep, "/")

            # Determine which category this file belongs to by checking
            # which known category name appears as a path component.
            parts = rel_path.split("/")
            category = None
            for cat in CATEGORIES:
                if cat in parts:
                    category = cat
                    break

            if category is None:
                print(f"  Skipping (no category match): {rel_path}")
                continue

            with open(filepath, "rb") as f:
                binary_data = f.read()

            b64_data = base64.b64encode(binary_data).decode("ascii")
            archives[category][rel_path] = b64_data
            total += 1
            print(f"  Encoded [{category}]: {rel_path}")

    # Write one JSON per category
    for cat in CATEGORIES:
        output_file = os.path.join(output_dir, f"{cat}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(archives[cat], f, indent=2, ensure_ascii=False)
        count = len(archives[cat])
        print(f"  -> {count} PNG file(s) written to '{output_file}'.")

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Encode PNG files in ablation_training into 3 JSON archives (one per category)."
    )
    parser.add_argument(
        "--input", "-i",
        default="./ablation_training",
        help="Base directory containing Fold_1 .. Fold_5 with category subfolders (default: ./ablation_training).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory where the three category JSON files are written (default: current directory).",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning '{input_dir}' for PNG files...")
    total = encode_ablation(input_dir, output_dir)

    print(f"\nAll done! {total} PNG file(s) encoded into {len(CATEGORIES)} JSON archive(s).")
    print(f"JSON archives written to '{output_dir}'.")


if __name__ == "__main__":
    main()
