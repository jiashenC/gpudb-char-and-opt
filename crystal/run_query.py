#!/usr/bin/env python3

import os
import time
import glob
import argparse
import subprocess as subp

from utility.profiler_logger import LOGGER


def modify_scale_factor(sf, is_opt_version=False):
    path = "crystal-opt_src" if is_opt_version else "crystal_src"

    file_line_list = []
    with open(f"./crystal/{path}/src/ssb/ssb_utils.h", "r") as f:
        for line in f.read().splitlines():
            if "#define SF" in line:
                line = "#define SF {}".format(sf)
            file_line_list.append(line)

    with open(f"./crystal/{path}/src/ssb/ssb_utils.h", "w") as f:
        f.write("\n".join(file_line_list))


def recompile(is_opt_version=False):
    path = "crystal-opt_src" if is_opt_version else "crystal_src"

    os.chdir(f"./crystal/{path}/")

    out, _ = subp.Popen(
        ["make"],
        stdin=subp.PIPE,
        stdout=subp.PIPE,
        stderr=subp.STDOUT,
    ).communicate()
    
    for qnum in [11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43]:
        if os.path.exists(f"./bin/ssb/q{qnum}"):
            os.remove(f"./bin/ssb/q{qnum}")
        out, _ = subp.Popen(
            [
                "make",
                f"bin/ssb/q{qnum}",
            ],
            stdin=subp.PIPE,
            stdout=subp.PIPE,
            stderr=subp.STDOUT,
        ).communicate()
    os.chdir("../../")
    print(out.decode("utf-8"))


def run(cmd, run):
    exec_cmd = f"{cmd} --t={run}"
    out, _ = subp.Popen(
        exec_cmd.split(" "),
        stdin=subp.PIPE,
        stdout=subp.PIPE,
        stderr=subp.STDOUT,
        env=os.environ.copy(),
    ).communicate()

    out = out.decode("utf-8")
    print(out)


def run_ncu(cmd, run):
    exec_cmd = f"{cmd} --t={run}"
    out, _ = subp.Popen(
        [
            "./utility/ncu_profiler.py",
            f"--bin='{exec_cmd}'",
        ],
        stdin=subp.PIPE,
        stdout=subp.PIPE,
        stderr=subp.STDOUT,
        env=os.environ.copy(),
    ).communicate()

    out = out.decode("utf-8")
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin", type=str, required=True, help="Binary executable."
    )
    parser.add_argument(
        "--profile-run", type=int, required=True, help="Profling trial."
    )
    parser.add_argument(
        "--sf", type=int, default=1, help="Scale factor for benchmarking."
    )
    parser.add_argument(
        "--ncu", default=False, action="store_true", help="Run Ncu profiling."
    )
    args = parser.parse_args()

    with open("./.log/profiler.log", "w") as f:
        f.close()

    LOGGER.debug("Delete old report")
    for name in glob.glob("gpudb-perf.*"):
        os.remove(name)

    is_opt_version =  "crystal-opt_src" in args.bin
    modify_scale_factor(args.sf, is_opt_version=is_opt_version)
    recompile(is_opt_version=is_opt_version)

    cmd = args.bin.strip("'")

    if args.ncu:
        run_ncu(cmd, 1)
    else:
        run(cmd, args.profile_run)


if __name__ == "__main__":
    main()
