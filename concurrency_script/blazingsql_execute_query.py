import os

os.environ["CONDA_PREFIX"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".local/miniconda3/envs/bsql",
)

import random
import argparse
import pandas as pd

from blazingsql import BlazingContext
from utility.profiler_logger import LOGGER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iter", type=int, default=10)
    parser.add_argument("--sf", type=int, default=2)
    args = parser.parse_args()

    config = {
        "BLAZING_LOGGING_DIRECTORY": "./.log/blazingsql_log/",
        "BLAZING_LOCAL_LOGGING_DIRECTORY": "./.log/blazingsql_log/",
    }

    bc = BlazingContext(config_options=config)

    table_path = "./blazingsql/table/ssb.txt"
    sf = args.sf

    # table drop and creation
    LOGGER.debug("Drop and create tables")
    table_list = []
    with open(table_path) as f:
        for line in f.read().splitlines():
            table_list.append(tuple(line.split(",")))

    for i, (tb, src) in enumerate(table_list):
        if tb in bc.list_tables():
            bc.drop_table(tb)
        LOGGER.debug(f"Create table: {tb} -- {src}")
        src = src.replace("[sfph]", str(sf))
        src_df = pd.read_parquet(src)
        bc.create_table(tb, src_df)

    cur_table = ",".join(bc.list_tables())
    LOGGER.debug(f"Table: {cur_table}")

    # execute sql
    LOGGER.debug("Parse SQL query")
    query_list = []
    for query_num in [11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43]:
        query = "q{}".format(query_num)
        query_path = "./blazingsql/sql/ssb_{}.sql".format(query)
        with open(query_path) as f:
            sql = f.read().replace("\n", " ").rstrip()
            sql_list = sql.split(";")[:-1]
            assert len(sql_list) == 1
            query_list.append(sql_list[0])

    random.shuffle(query_list)
    for _ in range(len(args.num_iter)):
        for query_str in query_list:
            _ = bc.sql(query_str)


if __name__ == "__main__":
    main()
