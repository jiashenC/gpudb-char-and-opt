#!/usr/bin/env python3

import os
import glob
import time
import argparse
import subprocess as subp

from utility.profiler_logger import LOGGER


def assemble_cmd(table, warm_sql, sql, sf, iter=1):
    py_interpreter = "./blazingsql/.local/miniconda3/envs/bsql/bin/python3"
    bin_cmd = f"{py_interpreter} blazingsql/execute_query.py --sql {sql} --table {table} --sf {sf} --iter {iter}"
    if warm_sql is not None:
        bin_cmd += f" --warm-sql {warm_sql}"
    LOGGER.debug(f"Assemble cmd {bin_cmd}")
    return bin_cmd


def run(table, warm_sql, sql, sf):
    bin_cmd = assemble_cmd(table, warm_sql, sql, sf, 5)

    LOGGER.debug("Execute SQL")
    out, _ = subp.Popen(
        bin_cmd.split(" "),
        stdout=subp.PIPE,
        stdin=subp.PIPE,
        stderr=subp.STDOUT,
        env=os.environ.copy(),
    ).communicate()
    out = out.decode("utf-8")
    print(out)


def run_ncu(table, warm_sql, sql, sf):
    skip_count = 0

    if warm_sql is not None:
        bin_cmd = assemble_cmd(table, None, warm_sql, sf)

        LOGGER.debug("Execute ncu warmup SQL")
        out, _ = subp.Popen(
            [
                "./utility/ncu_profiler.py",
                f"--bin='{bin_cmd}'",
                "--cmd-flags='--target-processes all'",
            ],
            stdout=subp.PIPE,
            stdin=subp.PIPE,
            stderr=subp.STDOUT,
            env=os.environ.copy(),
        ).communicate()

        out = out.decode("utf-8")
        skip_count = int(out.split("\n")[0])

    bin_cmd = assemble_cmd(table, warm_sql, sql, sf)

    LOGGER.debug("Execute ncu SQL")
    out, _ = subp.Popen(
        [
            "./utility/ncu_profiler.py",
            f"--bin='{bin_cmd}'",
            "--cmd-flags='--target-processes all'",
            "-s",
            f"{skip_count}",
        ],
        stdout=subp.PIPE,
        stdin=subp.PIPE,
        stderr=subp.STDOUT,
        env=os.environ.copy(),
    ).communicate()
    out = out.decode("utf-8")
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table", type=str, required=True, help="Table creation."
    )
    parser.add_argument("--sql", type=str, required=True, help="SQL query.")
    parser.add_argument(
        "--warm-sql", type=str, default="", help="Warmup SQL query."
    )
    parser.add_argument(
        "--sf", type=int, default=1, help="Scale factor for benchmarking."
    )
    parser.add_argument(
        "--ncu",
        default=False,
        action="store_true",
        help="Enable NSight compute profiling.",
    )
    args = parser.parse_args()

    with open("./.log/profiler.log", "w") as f:
        f.close()

    LOGGER.debug("Delete old report")
    for name in glob.glob("gpudb-perf.*"):
        os.remove(name)

    warm_sql = None if args.warm_sql == "" else args.warm_sql

    if args.ncu:
        run_ncu(args.table, warm_sql, args.sql, args.sf)
    else:
        run(args.table, warm_sql, args.sql, args.sf)


if __name__ == "__main__":
    main()
