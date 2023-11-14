#!/bin/bash

export HEAVYDB_ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
mkdir $HEAVYDB_ROOT/.local

# install heavydb dependencies
cd $HEAVYDB_ROOT/heavydb_src
./scripts/mapd-deps-ubuntu.sh

# env 
cd ..
source .local/mapd-deps.sh

# install heavydb
cd heavydb_src
mkdir build
cd build
cmake ..
make -j
