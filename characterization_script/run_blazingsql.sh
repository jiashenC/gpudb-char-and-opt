#!/bin/bash

mkdir -p tmp
mkdir -p tmp/res

env="CUDA_VISIBLE_DEVICES=$1"

all_queries=(11 12 13 21 22 23 31 32 33 34)

for sf_power in {0..4}; do

	sf=$((2 ** ${sf_power}))

	mkdir tmp/res/sf${sf}

	for qnum in ${all_queries[@]}; do
		cmd="${env} ./blazingsql/run_query.py --sql blazingsql/sql/ssb_q${qnum}.sql --table blazingsql/table/ssb.txt --sf ${sf} --ncu > output.txt"
		# echo $cmd > output.txt
		eval "$cmd"

		mkdir -p tmp/res/sf${sf}/q${qnum}
		mv gpudb-perf.ncu-rep tmp/res/sf${sf}/q${qnum}/
		mv output.txt tmp/res/sf${sf}/q${qnum}/
	done
done