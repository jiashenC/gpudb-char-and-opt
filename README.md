# Data Preparation

Crystal reads binary data. Other systems read parquet data. We use SSB script to generate dataset for benchmarking. Then, we convert them to appropriate data format, either binary or parquet format. Data will be stored in `data/storage` directory.
```
# Prepare environment for data generation
python3 -m venv data_venv
source data_venv/bin/activate
pip install pandas pyarrow fastparquet

# Data generation
mkdir data/storage
./data/generate_ssbm.py --sf 16
deactivate
```

# System Installation

Crystal implements one SSB query per file, so we need to compile each query manually. Example of compiling query 11.
```bash
cd crystal/crystal_src/
make && make bin/ssb/q11
cd ../../
```

HeavyDB requires installation from source.
```bash
git clone git@github.com/heavyai/heavydb.git
```
And then comment out `PREFIX=/usr/local/mapd-deps` in `heavydb/heavydb_src/scripts/mapd-deps-ubuntu.sh`. And add `PREFIX=$HEAVYDB_ROOT/.local/` after the commented out line.

For other systems (BlazingSQL and HeavyDB), we provide a bash script to install those systems. Their configuration files or compiled library will be stored in `<system>/.local` directory.
```bash
bash heavydb/install_heavydb.sh
bash blazingsql/install_blazingsql.sh
```

For HeavyDB, make sure it source the dependent libraries and also the its data directory is configured. 
```bash
source heavydb/.local/mapd-deps.sh
# Example of preparing data directory.
mkdir heavydb/heavydb_src/build/data
./heavydb/heavydb_src/build/bin/initheavy heavydb/heavydb_src/data
```

# Characterization - Resource (Section 4)

### End-to-End Time (Figure 5)

```bash
export PYTHONPATH="."
./crystal/run_query.py --bin="./crystal/crystal_src/bin/ssb/q11 --t=5" --profile-run 4 --sf 16
./heavydb/run_query.py --warm-sql ./heavydb/sql/ssb_q11_uncached.sql --sql ./heavydb/sql/ssb_q11_cached.sql --sf 16
./blazingsql/run_query.py --warm-sql blazingsql/sql/ssb_q11.sql --sql blazingsql/sql/ssb_q11.sql --table blazingsql/table/ssb.txt --sf 16
```

We rely on system standard output for end to end time, which can be found in the running output files.

### GPU Execution Time (Figure 5)

For micro-architectural statistics, scripts in `characteriztion_script` will run NSight Compute profiling and generate a report.
```bash
# We provide scripts to run all queries for all systems.
./characterization_script/run_crystal.sh
./characterization_script/run_heavydb.sh
./characterization_script/run_blazingsql.sh
# Command for profiling one query. Report statistics are outputted as standard output.
./crystal_src/run_query.py --bin="./crystal/crystal_src/bin/ssb/q11" --profile-run 1 --sf 16 --ncu
```
After running those scripts, profiling reports and its human readable outputs are saved in `tmp/res`.

All statistics can be exported to a human readable format by:
```bash
./stats/ncu_export.py --path <path to ncu report>
# Example to output q11 for Crystal, assuming reports gpudb-perf.ncu-rep is saved in tmp/res/sf1/q11.
./stats/ncu_export.py --path tmp/res/sf1/q11/
```

To process all ncu reports together and output to single file, all reports should be organzed follwoing `<root path>/<system>/<scale factor>/<query number>/gpudb-perf.ncu-rep`. Then script can be run to output all statistics together to a single file:
```bash
mkdir res
./stats/flush_ncu_csv.py --path <root path> --sf 16
```
Aggregate statistics will be saved inside `res` directory.

# Characterization - Bottleneck (Section 5)

Similar to gather `GPU Execution Time` metrics, similar steps can be used to gather all other micro-architectural statistics. Within the `flush_ncu_csv.py` file, there are different functions to flush different statistics, which include `# instructions`, `# bytes`, `roofline`, etc. Different files can be found in the `res` directory for relevant statistics.

* Figure 6 and 7 - `flush_roofline` and `flush_ai` function - `roofline_dram_<system>.txt`, `roofline_l2_<system>.txt`, and `ai.txt`
* Table 4 - `flush_top_kernel` function - `top_kernel_<query number>.txt`
* Figure 8 - `flush_inst` and `flush_bytes` function - `inst.txt` and `bytes.txt`
* Table 5 and Figure 9 `flush_stall` function - `stall_<query number>.txt`
* Table 6 - We manually inspect the human readable file

# Optimization 1 (Section 6)

All related code changes are in `crystal/crystal-opt_src`. We have made changes in `Makefile`, `crystal/load.cu`, `crystal/term.cu` and all query files. 

### Experiment Code Performance

Code performance can be compared by running both raw Crystal and Crystal-Opt libraries.

```bash
# Compile query
cd crystal/crystal_src && make clean && make && make bin/ssb/q11
cd crystal/crystal-opt_src && make clean && make && make bin/ssb/q11
# Run query
./crystal/crystal_src/bin/ssb/q11 --t=5
./crystal/crystal-opt_src/bin/ssb/q11 --t=5
```
Additionally, previously mentioned profiliing steps still work to gather all statistics.
```bash
# End to end
./crystal/run_query.py --bin="./crystal/crystal-opt_src/bin/ssb/q11 --t=5" --profile-run 4 --sf 16
# Micro-architecture 
./crystal/run_query.py --bin="./crystal/crystal-opt_src/bin/ssb/q11" --profile-run 1 --sf 16 --ncu
```

### Statistics for Comparison

* Figure 10 - the same script as `end-to-end time`
* Table 7 and Figure 11 - the same script as micro-architectural statistics

# Optimization 2 (Section 7)

All related code changes are in `concurrency_script`. We follow the NVIDIA instruction to modify MIG setting.

### Actual Throughput (Figure 14 top)

In this case, the degree of concurrency is controlled by number of actual partitions that GPU is configured to. For HeavyDB, because it is a client-server setup, the number of ports and the correct port number needs to be supplied in to script. At `heavydb` directory, there are a few configuration files, in which the server running port can be configured. And the ports in the script need to match with ports in the configuration. 

```bash
./concurrency_script/crystal_part_mig.py --sf 16 --iter 1000
./concurrency_script/heavydb_part_mig.py --sf 16 --iter 1000
./concurrency_script/blazingsql_part_mig.py --sf 16 --iter 1000
```

### Actual Throughput with MPS (Figure 15)

```bash
./concurrency_script/crystal_part_mps.py --sf 16 --iter 1000 --num-worker <degree of concurrency>
```

# Code Release

We release all system-related codes except TQP and the estimation model. TQP is still a close source library, so we do not include in this code release. The estimation model is currently protected by a Microsoft patent. 
