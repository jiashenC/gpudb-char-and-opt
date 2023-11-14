#!/bin/bash

export BLAZINGSQL_ROOT="./blazingsql"

# install miniconda first
$BLAZINGSQL_ROOT/.local/miniconda3/bin/python3 --version
retcode=$?
if [ $retcode -ne 0 ]; then
    wget -P $BLAZINGSQL_ROOT https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
    mkdir $BLAZINGSQL_ROOT/.local
    bash $BLAZINGSQL_ROOT/Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p $BLAZINGSQL_ROOT/.local/miniconda3/
    rm $BLAZINGSQL_ROOT/Miniconda3-py38_4.12.0-Linux-x86_64.sh
fi

export CONDA_ROOT=$BLAZINGSQL_ROOT/.local/miniconda3/bin/

# install blazingsql
$CONDA_ROOT/conda create --yes -n bsql
source $CONDA_ROOT/activate bsql
$CONDA_ROOT/conda install --yes -c blazingsql -c rapidsai -c nvidia -c conda-forge -c defaults blazingsql python=3.8 cudatoolkit=11.4