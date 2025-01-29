#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Jupyter notebook outputs")
    parser.add_argument(
        "-i",
        "--input_notebook",
        type=str,
        required=True,
        help="Input notebook file path",
    )
    parser.add_argument(
        "-o",
        "--output_notebook",
        type=str,
        required=True,
        help="Output notebook file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_fpath = Path(args.input_notebook)
    output_fpath = Path(args.output_notebook)

    print(f"Reading {input_fpath.name} ...")

    with open(input_fpath, "rt") as infile:
        notebook_contents = json.load(infile)

    print(f'Clearing output from {len(notebook_contents["cells"])} cells ...')
    for cell in notebook_contents["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []

    print(f"Writing cleaned notebook to {output_fpath.name} ...")
    with open(output_fpath, "wt") as outfile:
        json.dump(notebook_contents, outfile)

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
