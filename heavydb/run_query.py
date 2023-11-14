#!/usr/bin/env python3

import os
import time
import glob
import psutil
import signal
import argparse
import subprocess as subp

from utility.profiler_logger import LOGGER


def init_data():
    # database requires data storage initialization
    work_dir = "./heavydb/heavydb_src/build/"
    if not os.path.isdir(os.path.join(work_dir, "data")):
        os.mkdir(os.path.join(work_dir, "data"))
        subp.Popen(
            [
                os.path.join(work_dir, "bin/initheavy"),
                os.path.join(work_dir, "data"),
            ]
        ).wait()


def wait_server():
    LOGGER.debug("Wait for server to start ...")
    while True:
        client_p = subp.Popen(
            [
                "./heavydb/heavydb_src/build/bin/heavysql",
                "-p",
                "HyperInteractive",
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


def execute_sql(sql):
    LOGGER.debug(f"Execute {sql}")

    client_p = subp.Popen(
        [
            "./heavydb/heavydb_src/build/bin/heavysql",
            "-p",
            "HyperInteractive",
        ],
        stdout=subp.PIPE,
        stdin=subp.PIPE,
        stderr=subp.STDOUT,
    )

    sql = "\\timing\n" + sql
    out, _ = client_p.communicate(bytes(sql, encoding="utf-8"))

    print(out.decode("utf-8").split("\n")[-3])

    client_p.wait()

    # LOGGER.debug(out.decode("utf-8").split("\n")[-10])


def kill_heavydb_process():
    LOGGER.debug("Kill heavydb server")
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == "heavydb":
            pid = proc.info["pid"]
            os.kill(pid, signal.SIGTERM)
            break


def run(warm_sql, sql):
    LOGGER.debug("Launch heavydb server")
    server_p = subp.Popen(
        "./heavydb/heavydb_src/build/bin/heavydb --config ./heavydb/heavydb.conf".split(" "),
        stdout=subp.PIPE,
        stdin=subp.PIPE,
        stderr=subp.STDOUT,
        env=os.environ.copy(),
    )

    # warmup by executing sql
    if warm_sql is not None:
        # wait for server start
        wait_server()
        execute_sql(warm_sql)

    # execute sql
    for _ in range(5):
        execute_sql(sql)

    # kill the heavydb process with grace period
    time.sleep(10)
    kill_heavydb_process()

    out, _ = server_p.communicate()
    print(out.decode("utf-8"))


def run_ncu(warm_sql, sql):
    LOGGER.debug("NCU")

    LOGGER.debug("Launch heavydb server")
    server_p = subp.Popen(
        [
            "./utility/ncu_profiler.py",
            "--bin='./heavydb/heavydb_src/build/bin/heavydb --config ./heavydb/heavydb.conf'",
        ],
        stdout=subp.PIPE,
        stdin=subp.PIPE,
        stderr=subp.STDOUT,
        env=os.environ.copy(),
    )

    # warmup by executing sql
    if warm_sql is not None:
        # wait for server start
        wait_server()
        execute_sql(warm_sql)

        # kill the heavydb with grace period
        time.sleep(10)
        kill_heavydb_process()

        out, _ = server_p.communicate()
        out = out.decode("utf-8")
        skiped_count = int(out.split("\n")[0])

        LOGGER.debug("Launch heavydb server again after warmup")
        server_p = subp.Popen(
            [
                "./utility/ncu_profiler.py",
                "--bin='./heavydb/heavydb_src/build/bin/heavydb --config ./heavydb/heavydb.conf'",
                "-s",
                f"{skiped_count}",
            ],
            stdout=subp.PIPE,
            stdin=subp.PIPE,
            stderr=subp.STDOUT,
            env=os.environ.copy(),
        )

        # append warmup sql before
        sql = warm_sql + "\n" + sql

    # wait for warmup server again
    wait_server()

    # execute sql
    execute_sql(sql)

    # kill the heavydb process with grace period
    time.sleep(10)
    kill_heavydb_process()

    out, _ = server_p.communicate()
    print(out.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", type=str, required=True, help="SQL file.")
    parser.add_argument(
        "--warm-sql",
        default=None,
        type=str,
        help="SQL file to warmup system.",
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

    if args.warm_sql is not None:
        with open(args.warm_sql) as f:
            warm_sql = f.read().rstrip()
    else:
        warm_sql = None

    with open(args.sql) as f:
        sql = f.read().rstrip()

    # hacky way to rewrite scale factor to use for the benchmark
    sql = sql.replace("[sfph]", str(args.sf))
    if warm_sql is not None:
        warm_sql = warm_sql.replace("[sfph]", str(args.sf))

    with open("./.log/profiler.log", "w") as f:
        f.close()

    LOGGER.debug("Delete old report")
    for name in glob.glob("gpudb-perf.*"):
        os.remove(name)

    LOGGER.debug("Init data")
    init_data()

    if args.ncu:
        run_ncu(warm_sql, sql)
    else:
        run(warm_sql, sql)


if __name__ == "__main__":
    main()
