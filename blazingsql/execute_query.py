import os

os.environ["CONDA_PREFIX"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".local/miniconda3/envs/bsql",
)

import time
import pandas as pd
import argparse
import subprocess as subp

from blazingsql import BlazingContext

from utility.profiler_logger import LOGGER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table", type=str, required=True, help="Table creation."
    )
    parser.add_argument("--sql", type=str, required=True, help="SQL query.")
    parser.add_argument(
        "--warm-sql", type=str, default=None, help="Warmup SQL query."
    )
    parser.add_argument(
        "--sf", type=int, default=1, help="Scale factor for benchmarking."
    )
    parser.add_argument(
        "--iter", type=int, default=1, help="Number of iterations."
    )
    args = parser.parse_args()

    config = {
        "BLAZING_LOGGING_DIRECTORY": "./.log/blazingsql_log/",
        "BLAZING_LOCAL_LOGGING_DIRECTORY": "./.log/blazingsql_log/",
    }

    bc = BlazingContext(config_options=config)

    # table drop and creation
    LOGGER.debug("Drop and create tables")
    table_list = []
    with open(args.table) as f:
        for line in f.read().splitlines():
            table_list.append(tuple(line.split(",")))

    for i, (tb, src) in enumerate(table_list):
        if tb in bc.list_tables():
            bc.drop_table(tb)
        LOGGER.debug(f"Create table: {tb} -- {src}")
        src = src.replace("[sfph]", str(args.sf))
        src_df = pd.read_parquet(src)
        bc.create_table(tb, src_df)

    cur_table = ",".join(bc.list_tables())
    LOGGER.debug(f"Table: {cur_table}")

    if args.warm_sql is not None:
        LOGGER.debug("Parse warmup SQL query")
        with open(args.sql) as f:
            warm_sql = f.read().replace("\n", " ").rstrip()
            warm_sql_list = warm_sql.split(";")[:-1]

            for sql in warm_sql_list:
                LOGGER.debug(f"Warmup system with execution {sql}")
                _ = bc.sql(sql)

        time.sleep(1)

    # execute sql
    LOGGER.debug("Parse SQL query")
    with open(args.sql) as f:
        sql = f.read().replace("\n", " ").rstrip()
        sql_list = sql.split(";")[:-1]

    time.sleep(1)

    for _ in range(args.iter):
        st = time.perf_counter()
        for sql in sql_list:
            LOGGER.debug(f"Execute: {sql}")
            _ = bc.sql(sql)
        print(f"Execution time: {(time.perf_counter() - st) * 1000:.3f} ms")

    time.sleep(1)


if __name__ == "__main__":
    main()
