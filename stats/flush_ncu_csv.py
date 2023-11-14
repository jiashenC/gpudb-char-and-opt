#! /usr/bin/env python3

import os
import argparse

from collections import defaultdict

from report_parser.ncu_parser import NcuParser
from utility.counter_config import *


SYS_2_SYSLABEL = {
    "crystal": "Crystal",
    "crystal-opt": "Crystal-Opt",
    "heavydb": "HeavyDB",
    "bsql": "BlazingSQL",
    "tqp": "TQP",
}


def append_data(stats_list, data, data_idx, idx):
    if idx == 0:
        stats_list.append([])
    stats_list[data_idx].append(data)


def generate_file_path(root_path, sys, sf, query):
    file_path = os.path.join(
        root_path, sys, "sf{}".format(sf), query
    )
    return file_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf", type=int, default=-1, help="Scale factor.")
    parser.add_argument("--path", type=str, default=".", help="Root path to all ncu reports.")
    args = parser.parse_args()

    # sys_list = ["crystal", "heavydb", "bsql", "tqp"]
    sys_list = ["crystal", "heavydb", "bsql"]
    query_list = [
        "q11",
        "q12",
        "q13",
        "q21",
        "q22",
        "q23",
        "q31",
        "q32",
        "q33",
        "q34",
        "q41",
        "q42",
        "q43",
    ]

    root_path = args.path

    flush_inst(root_path, sys_list, query_list, args)
    flush_bytes(root_path, sys_list, query_list, args)
    flush_ai(root_path, sys_list, query_list, args)
    flush_stall(root_path, sys_list, query_list, args)
    flush_roofline(root_path, sys_list, query_list, args)
    flush_top_kernel(root_path, sys_list, query_list, args)


def flush_inst(root_path, sys_list, query_list, args):
    aggre_res = [[] for _ in range(len(sys_list))]
    with open("./res/inst.txt", "w") as f:
        f.write("Crystal,HeavyDB,BlazingSQL,TQP\n")
        for query in query_list:
            for i, sys in enumerate(sys_list):
                file_path = generate_file_path(root_path, sys, args.sf, query)
                if not os.path.exists(file_path):
                    f.write("0,")
                    continue

                ncu_parser = NcuParser(file_path)
                kernel_list = ncu_parser.get_kernel_list()

                tot_inst = 0
                for kernel in kernel_list:
                    tot_inst += (
                        ncu_parser.get_value(
                            kernel,
                            "smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed",
                        )
                        * ncu_parser.get_value(
                            kernel, "smsp__cycles_elapsed.avg.per_second"
                        )
                        * ncu_parser.get_value(
                            kernel, "gpu__time_duration.sum"
                        )
                    )
                aggre_res[i].append(tot_inst)
                f.write("{},".format(int(tot_inst)))

            f.write("Q{}\n".format(query[1:]))

        for res in aggre_res:
            f.write("{},".format(sum(res) / len(res)))
        f.write("Avg.\n")


def flush_bytes(root_path, sys_list, query_list, args):
    aggre_res = [[] for _ in range(len(sys_list))]
    with open("./res/bytes.txt", "w") as f:
        f.write("Crystal,HeavyDB,BlazingSQL,TQP\n")
        for query in query_list:
            for i, sys in enumerate(sys_list):
                file_path = generate_file_path(root_path, sys, args.sf, query)
                if not os.path.exists(file_path):
                    f.write("0,")
                    continue

                ncu_parser = NcuParser(file_path)
                kernel_list = ncu_parser.get_kernel_list()

                tot_bytes = 0
                for kernel in kernel_list:
                    tot_bytes += ncu_parser.get_value(
                        kernel, "dram__bytes_read.sum"
                    )
                aggre_res[i].append(tot_bytes)
                f.write("{},".format(int(tot_bytes)))

            f.write("Q{}\n".format(query[1:]))

        for res in aggre_res:
            f.write("{},".format(sum(res) / len(res)))
        f.write("Avg.\n")


def flush_ai(root_path, sys_list, query_list, args):
    aggre_res1 = [[] for _ in range(len(sys_list))]
    aggre_res2 = [[] for _ in range(len(sys_list))]
    with open("./res/ai.txt", "w") as f:
        f.write("Crystal,HeavyDB,BlazingSQL,TQP\n")
        for query in query_list:
            for i, sys in enumerate(sys_list):
                file_path = generate_file_path(root_path, sys, args.sf, query)
                if not os.path.exists(file_path):
                    f.write("0,")
                    continue

                ncu_parser = NcuParser(file_path)
                kernel_list = ncu_parser.get_kernel_list()

                tot_bytes = 0
                for kernel in kernel_list:
                    tot_bytes += ncu_parser.get_value(
                        kernel, "dram__bytes_read.sum"
                    )
                aggre_res1[i].append(tot_bytes)

                tot_inst = 0
                for kernel in kernel_list:
                    tot_inst += (
                        ncu_parser.get_value(
                            kernel,
                            "smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed",
                        )
                        * ncu_parser.get_value(
                            kernel, "smsp__cycles_elapsed.avg.per_second"
                        )
                        * ncu_parser.get_value(
                            kernel, "gpu__time_duration.sum"
                        )
                    )
                aggre_res2[i].append(tot_inst)

                f.write("{},".format(tot_inst / tot_bytes))

            f.write("Q{}\n".format(query[1:]))

        for i, res in enumerate(aggre_res2):
            f.write("{},".format(sum(res) / sum(aggre_res1[i])))
        f.write("Avg.\n")


def flush_stall(root_path, sys_list, query_list, args):
    counter_list = [
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    ]

    counter_to_label = {
        "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio": "Mem WB",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "Branch",
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "Compute",
        "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio": "Mem Queue Full",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "Mem LD",
    }

    per_sys_per_query_avg_stall_aggre = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0.0))
    )
    for sys in sys_list:
        for query in query_list:
            file_path = generate_file_path(root_path, sys, args.sf, query)
            if not os.path.exists(file_path):
                continue

            stall_aggre = defaultdict(lambda: 0.0)

            ncu_parser = NcuParser(file_path)
            kernel_list = ncu_parser.get_kernel_list()

            # aggregate stall information across multiple kernels
            tot_inst = 0
            for kernel in kernel_list:
                inst = ncu_parser.get_value(kernel, "smsp__inst_executed.sum")
                tot_inst += inst
                for counter in counter_list:
                    stall_aggre[counter] += inst * ncu_parser.get_value(
                        kernel, counter
                    )

            # insert per query and per system stall information
            for counter in counter_list:
                per_sys_per_query_avg_stall_aggre[sys][query][counter] += (
                    stall_aggre[counter] / tot_inst
                )

        with open("./res/stall_{}.txt".format(sys), "w+") as f:
            for counter in counter_list:
                stall_list = []
                for query in query_list:
                    if query not in per_sys_per_query_avg_stall_aggre[sys]:
                        continue
                    total_stall = sum(
                        per_sys_per_query_avg_stall_aggre[sys][query].values()
                    )
                    stall_list.append(
                        per_sys_per_query_avg_stall_aggre[sys][query][counter]
                        / total_stall
                    )
                stall_list = sorted(stall_list)
                f.write(",".join([str(stall) for stall in stall_list]))
                f.write(",{}\n".format(counter_to_label[counter]))

    for query in query_list:
        with open("./res/stall_q{}.txt".format(query[1:]), "w+") as f:
            f.write(
                ",".join(
                    [counter_to_label[counter] for counter in counter_list]
                )
            )
            f.write("\n")
            for sys in sys_list:
                for counter in counter_list:
                    f.write(
                        "{},".format(
                            per_sys_per_query_avg_stall_aggre[sys][query][
                                counter
                            ]
                        )
                    )
                f.write("{}\n".format(SYS_2_SYSLABEL[sys]))


def flush_top_kernel(root_path, sys_list, query_list, args):
    for query in query_list:
        with open("./res/top_kernel_{}.txt".format(query), "w") as f:
            for sys in sys_list:
                file_path = generate_file_path(root_path, sys, args.sf, query)

                if not os.path.exists(file_path):
                    for _ in range(5):
                        f.write("None:0,")
                    f.write("{}\n".format(sys))
                    continue

                ncu_parser = NcuParser(file_path)
                kernel_list = ncu_parser.get_kernel_list()

                kernel_aggre = defaultdict(lambda: 0.0)

                for kernel in kernel_list:
                    kernel_name = "_".join(kernel.split("_")[:-1])
                    kernel_aggre[kernel_name] += ncu_parser.get_value(
                        kernel, "gpu__time_duration.sum"
                    )

                kernel_time_list = [
                    (kernel_name, kernel_time)
                    for kernel_name, kernel_time in kernel_aggre.items()
                ]
                top_kernel_list = sorted(
                    kernel_time_list, key=lambda x: x[1], reverse=True
                )[:3]

                tot_kernel_time = sum(kernel_aggre.values())
                for kernel_name, _ in top_kernel_list:
                    f.write(
                        "{}:{},".format(
                            kernel_name,
                            kernel_aggre[kernel_name] / tot_kernel_time,
                            # kernel_aggre[kernel_name] * 1000,
                        )
                    )
                f.write("{}\n".format(sys))
            f.flush()


def flush_roofline(root_path, sys_list, query_list, args):
    for sys in sys_list:

        dram_f = open("./res/roofline_dram_{}.txt".format(sys), "w")
        l2_f = open("./res/roofline_l2_{}.txt".format(sys), "w")

        dram_f.write("{}\n".format(SYS_2_SYSLABEL[sys]))
        l2_f.write("{}\n".format(SYS_2_SYSLABEL[sys]))

        for query in query_list:
            file_path = generate_file_path(root_path, sys, args.sf, query)
            if not os.path.exists(file_path):
                continue

            ncu_parser = NcuParser(file_path)
            kernel_list = ncu_parser.get_kernel_list()

            tot_kernel_time, tot_l2_req, tot_bytes, tot_inst = 0, 0, 0, 0

            for kernel in kernel_list:
                tot_kernel_time += ncu_parser.get_value(
                    kernel, "gpu__time_duration.sum"
                )
                tot_l2_req += ncu_parser.get_value(
                    kernel, "lts__t_requests_srcunit_tex_op_read.sum"
                )
                tot_bytes += ncu_parser.get_value(
                    kernel, "dram__bytes_read.sum"
                )
                tot_inst += (
                    ncu_parser.get_value(
                        kernel,
                        "smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed",
                    )
                    * ncu_parser.get_value(
                        kernel, "smsp__cycles_elapsed.avg.per_second"
                    )
                    * ncu_parser.get_value(kernel, "gpu__time_duration.sum")
                )

            tot_l2_bytes = tot_l2_req * 128

            dram_f.write(
                "{},{}\n".format(
                    tot_inst / tot_bytes,
                    tot_inst / tot_kernel_time / 1000000000,
                )
            )
            l2_f.write(
                "{},{}\n".format(
                    tot_inst / tot_l2_bytes,
                    tot_inst / tot_kernel_time / 1000000000,
                )
            )


if __name__ == "__main__":
    main()
