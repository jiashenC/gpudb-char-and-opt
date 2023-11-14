#! /usr/bin/env python3

import os
import pandas as pd
import subprocess as subp


def main():
    data_dir = "./data/storage"
    for path in os.listdir(data_dir):
        if not ".txt" in path:
            continue

        name = path.split(".")[0]
        print(name, "...")
        p = subp.Popen(
            [
                "./data/convert_to_parquet.py",
                "--in-path",
                os.path.join(data_dir, name + ".txt"),
                "--out-path",
                os.path.join(data_dir, name + ".parquet"),
            ],
            stdin=subp.PIPE,
            stdout=subp.PIPE,
            stderr=subp.STDOUT,
        )
        print(p.communicate()[0].decode("utf-8"))


if __name__ == "__main__":
    main()
