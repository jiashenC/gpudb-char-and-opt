#!/bin/bash

mkdir -p tmp
mkdir -p tmp/res

# compile_query() {
# 	cd crystal/crystal_src
# 	make clean
# 	make
# 	make bin/ssb/q11 && make bin/ssb/q12 && make bin/ssb/q13 && make bin/ssb/q21 && make bin/ssb/q22 && make bin/ssb/q23 && make bin/ssb/q31 && make bin/ssb/q32 && make bin/ssb/q33 && make bin/ssb/q34 && make bin/ssb/q41 && make bin/ssb/q42 && make bin/ssb/q43

# 	cd ../../
# }

all_queries=(11 12 13 21 22 23 31 32 33 34 41 42 43)

env="CUDA_VISIBLE_DEVICES=$1"

for sf_power in {0..0}; do

	sf=$((2 ** ${sf_power}))

	# sed "10d" crystal/crystal_src/src/ssb/ssb_utils.h >.tmp
	# sed "9 a\
	# 	#define SF ${sf}" .tmp >crystal/crystal_src/src/ssb/ssb_utils.h

	# compile_query

	mkdir -p tmp/res/sf${sf}

	for qnum in ${all_queries[@]}; do

		mkdir -p tmp/res/sf${sf}/q${qnum}

		cmd1="${env} ./crystal/run_query.py --ncu --bin='./crystal/crystal_src/bin/ssb/q"
		cmd2="$qnum"
		cmd3="' --profile-run 1 --sf ${sf} > output.txt"
		cmd="$cmd1$cmd2$cmd3"
		eval "$cmd"

		mv gpudb-perf.ncu-rep tmp/res/sf${sf}/q${qnum}/
		mv output.txt tmp/res/sf${sf}/q${qnum}/
	done
done
