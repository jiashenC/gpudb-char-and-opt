#!/bin/bash

mkdir -p tmp
mkdir -p tmp/res

env="CUDA_VISIBLE_DEVICES=$1"

all_queries=(11 12 13 21 22 23 31 32 33 34 41 42 43)

for sf_power in {0..4}; do

	sf=$((2 ** ${sf_power}))

	mkdir -p tmp/res/sf${sf}

	for qnum in ${all_queries[@]}; do

		cmd="./heavydb/run_query.py --sql ./heavydb/sql/ssb_q${qnum}_uncached.sql --sf ${sf} --ncu > output.txt"
		eval "$cmd"

		mkdir -p tmp/res/sf${sf}/q${qnum}
		mv gpudb-perf.ncu-rep tmp/res/sf${sf}/q${qnum}/
		mv output.txt tmp/res/sf${sf}/q${qnum}/
	done
done