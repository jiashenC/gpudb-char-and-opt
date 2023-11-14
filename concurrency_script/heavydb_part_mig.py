#!/usr/bin/env python3

import os
import time
import psutil
import signal
import argparse
import subprocess as subp

from threading import Thread
from collections import defaultdict

from report_parser.ncu_parser import NcuParser
from utility.profiler_logger import LOGGER


def launch_cmd(cmd):
    out, _ = subp.Popen(
        cmd.split(" "),
        stdin=subp.PIPE,
        stdout=subp.PIPE,
        stderr=subp.STDOUT,
    ).communicate()
    return out.decode("utf-8")


def enable_mig():
    launch_cmd("sudo nvidia-smi -mig 1")
    launch_cmd("sudo nvidia-smi mig -cgi 9,9 -C")
    launch_cmd("sudo nvidia-smi -lgc 1410")


def disable_mig():
    launch_cmd("sudo nvidia-smi mig -dci -ci 0 -gi 1")
    launch_cmd("sudo nvidia-smi mig -dci -ci 0 -gi 2")
    launch_cmd("sudo nvidia-smi mig -dgi")
    launch_cmd("sudo nvidia-smi -mig 0")
    launch_cmd("sudo nvidia-smi -lgc 1410")


def get_device():
    device_list = launch_cmd("nvidia-smi -L").strip("\n").split("\n")
    for i, device in enumerate(device_list):
        device = device.strip(")")
        device = device.split(":")[-1].strip(" ")
        device_list[i] = device
    return device_list


def wait_server(port):
    LOGGER.debug("Wait for heavydb server to start ...")
    while True:
        client_p = subp.Popen(
            [
                "./heavydb/heavydb_src/build/bin/heavysql",
                "-p",
                "HyperInteractive",
                "--port",
                str(port),
            ],
            stdout=subp.PIPE,
            stdin=subp.PIPE,
            stderr=subp.STDOUT,
        )
        client_p.communicate()
        retcode = client_p.returncode
        client_p.kill()
        if retcode == 0:
            break


def start_server(config_path):
    LOGGER.debug("Start heavydb server")
    server_p = subp.Popen(
        [
            "./heavydb/heavydb_src/build/bin/heavydb",
            "--config",
            config_path,
        ],
        stdin=subp.PIPE,
        stdout=subp.PIPE,
        stderr=subp.STDOUT,
        env=os.environ.copy(),
    )

    return server_p


def stop_server():
    LOGGER.debug("Stop heavydb server")
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == "heavydb":
            pid = proc.info["pid"]
            os.kill(pid, signal.SIGTERM)


def create_table(port):
    LOGGER.debug("Heavydb creates table")

    with open("./heavydb/sql/ssb_create_table.sql") as f:
        sql = f.read().rstrip()

    return execute_sql(sql, port)


def copy_data(sf, port):
    LOGGER.debug("Heavydb copies data")

    with open("./heavydb/sql/ssb_load_table.sql") as f:
        sql = f.read().rstrip()
    sql = sql.replace("[sfph]", str(sf))

    return execute_sql(sql, port)


def execute_sql(sql, port):
    st = time.perf_counter() * 1000

    client_p = subp.Popen(
        [
            "./heavydb/heavydb_src/build/bin/heavysql",
            "-p",
            "HyperInteractive",
            "--port",
            str(port),
        ],
        stdout=subp.PIPE,
        stdin=subp.PIPE,
        stderr=subp.STDOUT,
    )

    client_p.communicate(bytes(sql, encoding="utf-8"))
    client_p.wait()

    t = time.perf_counter() * 1000 - st
    return t


def task_client(port, task_list, iter):
    tb_creation_t, data_load_t, execute_t, total_t = (
        0,
        0,
        0,
        time.perf_counter() * 1000.0,
    )

    prev_sf = -1
    query_sql_list = []

    for sf, sql in task_list:
        if sf != prev_sf:
            tb_creation_t += create_table(port)
            data_load_t += copy_data(sf, port)
            prev_sf = sf
            if len(query_sql_list) != 0:
                execute_t += execute_sql("\n".join(query_sql_list), port)
                query_sql_list = []

        for _ in range(iter):
            query_sql_list.append(sql)

    if len(query_sql_list) != 0:
        execute_t += execute_sql("\n".join(query_sql_list), port)
        query_sql_list = []

    total_t = time.perf_counter() * 1000.0 - total_t

    print(f"[{port}] Total: {total_t:.3f} ms")
    print(
        f"[{port}] Execution: {execute_t:.3f} ms, Table Creation: {tb_creation_t:.3f} ms, Data Copy: {data_load_t:.3f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iter",
        type=int,
        required=True,
        help="Number of iterations to run SSB benchmarks.",
    )
    parser.add_argument(
        "--sf-list",
        type=int,
        required=True,
        nargs="+",
        help="Max scale factor to run.",
    )
    args = parser.parse_args()

    with open("./.log/profiler.log", "w") as f:
        f.close()

    query_num_list = [11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43]

    server_port_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000]

    per_server_query = defaultdict(lambda: [])

    server_alloc_idx = 0
    for sf in args.sf_list:
        for qnum in query_num_list:
            with open(f"./heavydb/sql/ssb_q{qnum}_cached.sql") as f:
                sql = f.read().rstrip()
            server_idx = server_alloc_idx % len(server_port_list)
            per_server_query[server_port_list[server_idx]].append((sf, sql))
            server_alloc_idx += 1

    mig_list = [d for d in get_device() if "MIG" in d]

    for i in range(len(server_port_list)):
        os.environ["CUDA_VISIBLE_DEVICES"] = mig_list[i]
        start_server(f"./heavydb/heavydb_part{i}.conf")

    # concurrently wait for servers
    thread_pool = []
    for p in server_port_list:
        thread_pool.append(Thread(target=wait_server, args=(p,)))
    for t in thread_pool:
        t.start()
    for t in thread_pool:
        t.join()

    total_t = time.perf_counter() * 1000

    # concurrently start clients
    thread_pool = []
    for port, task_list in per_server_query.items():
        thread_pool.append(
            Thread(target=task_client, args=(port, task_list, args.iter))
        )
    for t in thread_pool:
        t.start()
    for t in thread_pool:
        t.join()

    total_t = time.perf_counter() * 1000 - total_t

    stop_server()

    print(f"Total: {total_t:.3f} ms")


if __name__ == "__main__":
    main()
