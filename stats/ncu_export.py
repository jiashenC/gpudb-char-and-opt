#!/usr/bin/env python3

import argparse

from report_parser.ncu_parser import NcuParser
from utility.counter_config import *


def print_res(kn, metric_to_label, parser):
    for per_res in parser.gen_res(kn, list(metric_to_label.keys())):
        print(metric_to_label[per_res[0]], end=",")
        print(per_res[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", required=True, type=str, help="Path to parse ncu report."
    )
    args = parser.parse_args()

    ncu_par = NcuParser(path=args.path)

    kernel_list = ncu_par.get_kernel_list()

    for kn in kernel_list:
        print("-----------------------------------------------------")
        print("Kernel name:", kn)
        print("------------------------------------")
        print_res(kn, metric_sol(), ncu_par)
        print("------------------------------------")
        print_res(kn, metric_roofline(), ncu_par)
        print("------------------------------------")
        print_res(kn, metric_occupancy(), ncu_par)
        print("------------------------------------")
        print_res(kn, metric_compute(), ncu_par)
        print("------------------------------------")
        print_res(kn, metric_memory(), ncu_par)
        print("------------------------------------")
        print_res(kn, metric_launch(), ncu_par)
        print("------------------------------------")
        print_res(kn, metric_warp(), ncu_par)
        print("------------------------------------")
        print_res(kn, metric_detail_warp(), ncu_par)
        print("-----------------------------------------------------")
        print_res(kn, metric_inst(), ncu_par)
        print("-----------------------------------------------------")


if __name__ == "__main__":
    main()
