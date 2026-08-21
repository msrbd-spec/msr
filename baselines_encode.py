#!/usr/bin/env python3
"""
Encode all PNG files in the baselines directory tree into separate JSON archives.

Two JSON archives are generated per model folder (e.g. ConvNeXtV2-Tiny),
split by run type:
  - {model}_Run1_archive.json   -> all PNGs under FL_Run1_Uniform/ folders
  - {model}_Run2_archive.json   -> all PNGs under FL_Run2_Heterogeneous/ folders

The folder structure within each model is preserved by using relative file
paths as keys in the JSON object. PNG binary data is base64-encoded (lossless).
Only .png files are encoded; .pt and .json files are skipped.

Usage:
    python baselines_encode.py
    python baselines_encode.py --input . --output .
"""

import argparse
import base64
import json
import os
import sys


# The four model folders to encode separately
MODEL_FOLDERS = [
    "ConvNeXtV2-Tiny",
    "EfficientNetV2-S",
    "MobileNetV2",
    "ResNet50",
]

# The two run subfolders that split each model's PNGs
RUN_FOLDERS = [
    ("Run1", "FL_Run1_Uniform"),
    ("Run2", "FL_Run2_Heterogeneous"),
]


def encode_run_pngs(model_dir: str, run_subfolder: str, output_file: str) -> int:
    """Walk model_dir for PNGs inside run_subfolder, base64-encode, write to output_file."""
    archive = {}
    count = 0

    for root, dirs, files in os.walk(model_dir):
        # Only process directories that contain the run_subfolder in their path
        if run_subfolder not in root:
            continue
        for filename in sorted(files):
            if not filename.lower().endswith(".png"):
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, model_dir)
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

    print(f"  -> {count} PNG file(s) encoded into '{output_file}'.\n")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Encode all PNG files in the baselines directory into separate JSON archives (two per model)."
    )
    parser.add_argument(
        "--input", "-i",
        default=".",
        help="Base directory containing the 4 model folders (default: current directory).",
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory for JSON archive files (default: current directory).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    total_count = 0
    num_archives = 0
    for model_name in MODEL_FOLDERS:
        model_dir = os.path.join(args.input, model_name)
        if not os.path.isdir(model_dir):
            print(f"Warning: Model folder '{model_name}' not found, skipping.")
            continue

        for run_label, run_subfolder in RUN_FOLDERS:
            output_file = os.path.join(args.output, f"{model_name}_{run_label}_archive.json")
            print(f"Scanning '{model_name}/{run_subfolder}' for PNG files...")
            total_count += encode_run_pngs(model_dir, run_subfolder, output_file)
            num_archives += 1

    print(f"Done! {total_count} PNG file(s) total encoded into {num_archives} archive(s).")


if __name__ == "__main__":
    main()
