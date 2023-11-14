#! /usr/bin/env python3

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-path", type=str, required=True, help="Input file path."
    )
    parser.add_argument(
        "--out-path", type=str, required=True, help="Output file path."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df.to_parquet(args.out_path)


if __name__ == "__main__":
    main()
