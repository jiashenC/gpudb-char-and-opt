#!/usr/bin/env python3

import os
import time
import argparse
import subprocess as subp

from threading import Thread


def modify_scale_factor(sf):
    file_line_list = []
    with open("./crystal/crystal_src/src/ssb/ssb_utils.h", "r") as f:
        for line in f.read().splitlines():
            if "#define SF" in line:
                line = "#define SF {}".format(sf)
            file_line_list.append(line)

    with open("./crystal/crystal_src/src/ssb/ssb_utils.h", "w") as f:
        f.write("\n".join(file_line_list))


def recompile():
    os.chdir("./crystal/crystal_src/")
    if os.path.exists("./bin/ssb/all"):
        os.remove("./bin/ssb/all")
    out, _ = subp.Popen(
        [
            "make",
            "bin/ssb/all",
        ],
        stdin=subp.PIPE,
        stdout=subp.PIPE,
        stderr=subp.STDOUT,
    ).communicate()
    os.chdir("../../")
    print(out.decode("utf-8"))


def launch_cmd(cmd):
    out, _ = subp.Popen(
        cmd.split(" "),
        stdin=subp.PIPE,
        stdout=subp.PIPE,
        stderr=subp.STDOUT,
    ).communicate()
    return out.decode("utf-8")


def start_server(num_iter):
    subp.Popen(
        [
            "./crystal/crystal_src/bin/ssb/all",
            "--t={}".format(num_iter),
        ],
        env=os.environ.copy(),
    ).wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iter",
        type=int,
        required=True,
        help="Number of iterations to run SSB benchmarks.",
    )
    parser.add_argument(
        "--sf",
        type=int,
        default=2,
        help="Scale factor to run SSB benchmarks.",
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=1,
        help="Number of workers."
    )
    args = parser.parse_args()

    modify_scale_factor(args.sf)
    recompile()

    launch_cmd("nvidia-cuda-mps-control -d")

    total_t = time.perf_counter() * 1000

    thread_pool = []
    for _ in range(args.num_worker):
        thread_pool.append(Thread(target=start_server, args=(args.iter,)))
    for t in thread_pool:
        t.start()
    for t in thread_pool:
        t.join()

    total_t = time.perf_counter() * 1000 - total_t
    print(f"Total: {total_t:.3f} ms")

    launch_cmd("echo quit | nvidia-cuda-mps-control")


if __name__ == "__main__":
    main()