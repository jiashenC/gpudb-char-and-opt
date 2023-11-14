#!/usr/bin/env python3

import os
import time
import argparse
import subprocess as subp

from report_parser.ncu_parser import NcuParser
from utility.profiler_logger import LOGGER
from utility.counter_config import *


ARGS = None


def append_metric(cmd, metric_list):
    metric_str = ",".join(metric_list)
    cmd += f"--metrics {metric_str} --apply-rules yes --clock-control none "
    return cmd


def wait_report():
    LOGGER.debug("Wait for ncu report")
    while True:
        if os.path.isfile("./gpudb-perf.ncu-rep"):
            break
        time.sleep(0.2)


def run(cmd):
    binary = ARGS.bin.strip("'")
    cmd += f"-f -o gpudb-perf {binary}"

    LOGGER.debug(f"Profiler runs {cmd}")

    cmd_list = cmd.split()
    out, _ = subp.Popen(
        cmd_list,
        stdin=subp.PIPE,
        stdout=subp.PIPE,
        stderr=subp.STDOUT,
        env=os.environ.copy(),
    ).communicate()

    out = out.decode("utf-8")
    LOGGER.debug(out)

    wait_report()


def print_res(kn, metric_to_label, parser):
    for per_res in parser.gen_res(kn, list(metric_to_label.keys())):
        print(metric_to_label[per_res[0]], end=",")
        print(per_res[1])


def main():
    LOGGER.debug("Profiler starts ...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin", required=True, type=str, help="Profiled binary."
    )
    parser.add_argument(
        "-s", default=-1, type=int, help="Number of kernels to skip."
    )
    parser.add_argument(
        "-c", default=-1, type=int, help="Number of kernels to profile."
    )
    parser.add_argument(
        "--cmd-flags", default="", type=str, help="Extra flags send to ncu."
    )

    global ARGS
    ARGS = parser.parse_args()

    cmd = "ncu "
    if ARGS.s != -1:
        cmd += f"-s {ARGS.s} "
    if ARGS.c != -1:
        cmd += f"-c {ARGS.c} "
    cmd_flags = ARGS.cmd_flags.strip("'")
    cmd += f"{cmd_flags} "

    # append metrics flags
    metric_list = []
    metric_list += list(metric_sol().keys())
    metric_list += list(metric_roofline().keys())
    metric_list += list(metric_occupancy().keys())
    metric_list += list(metric_compute().keys())
    metric_list += list(metric_memory().keys())
    metric_list += list(metric_launch().keys())
    metric_list += list(metric_warp().keys())
    metric_list += list(metric_detail_warp().keys())
    metric_list += list(metric_inst().keys())

    cmd = append_metric(cmd, metric_list)

    # actual run
    run(cmd)

    # parse ncu csv data
    parser = NcuParser()

    # print all metrics in per-kernel base
    kernel_list = parser.get_kernel_list()
    kernel_list_str = ",".join(kernel_list)
    LOGGER.debug(f"Kernel list: {kernel_list_str}")

    print(f"{len(kernel_list)}")
    for kn in kernel_list:
        print("-----------------------------------------------------")
        print("Kernel name:", kn)
        print("------------------------------------")
        print_res(kn, metric_sol(), parser)
        print("------------------------------------")
        print_res(kn, metric_roofline(), parser)
        print("------------------------------------")
        print_res(kn, metric_occupancy(), parser)
        print("------------------------------------")
        print_res(kn, metric_compute(), parser)
        print("------------------------------------")
        print_res(kn, metric_memory(), parser)
        print("------------------------------------")
        print_res(kn, metric_launch(), parser)
        print("------------------------------------")
        print_res(kn, metric_warp(), parser)
        print("------------------------------------")
        print_res(kn, metric_detail_warp(), parser)
        print("-----------------------------------------------------")
        print_res(kn, metric_inst(), parser)
        print("-----------------------------------------------------")


if __name__ == "__main__":
    main()
