#!/bin/sh -e

# SPDX-License-Identifier: MIT

cmd=$(basename $0)
SLAB_ROOT=../..
. $SLAB_ROOT/utils/utils.sh

#######################################################################
# Defaults

compression=0
njobs=1

#######################################################################

show_usage() {
    echo "usage: $cmd [options] <dataset> <config>"
}

show_help() {
    show_usage
    echo ""
    echo "Extract speech features for <dataset> into a HDF5 archive."
    echo ""
    echo "  -h              show this help message"
    echo "  -c C            compression level from 0 to 9 (default: $compression)"
    echo "  -n N            use N parallel jobs (default: $njobs)"
}

while :; do
    case "$1" in
        -c) compression=$2; shift;;
        -h) show_help; exit 0;;
        -n) njobs=$2; shift;;
        -*)
            echo "unknown option: $1. Try '$0 -h' for more informations."  1>&2
            echo "" 1>&2
            show_usage 1>&2
            exit 1;;
        *) break;;
    esac
    shift
done

if [ $# -ne 2 ]; then
    show_usage 1>&2
    exit 1
fi

datasetname=$1
config=$2

assert_is_dataset $datasetname
datasetdir=$SLAB_DATASETS/$datasetname
odir=$SLAB_FEATURES/$datasetname
mkdir -p $odir

# The features name correspond to the name of configuration file
# without the extension.
bname=$(basename $config)
feaname=${bname%%.*}

out=$odir/$feaname.h5
scp=$datasetdir/wav.scp
assert_not_missing $scp

#if [ -f $out ]; then
#    echo "The features are already extracted ($out)."
#    exit 0
#fi

echo extracting features from $scp to $out

tmp=$(mktemp -d -p $odir)
trap 'rm -fr "$tmp"; trap - EXIT; exit' EXIT INT HUP
cwd=$(pwd)
cd $tmp
split -n l/$njobs $scp --numeric-suffixes=1
cd $cwd

. parallel_env.sh
$parallel_cmd julia --project $SLAB_ROOT/recipes/features/scripts/features.jl \
    -c $compression \
    conf/mfcc_d_dd_16kHz.toml \
    $tmp/x*\$SGE_TASK_ID \
    $tmp/\$SGE_TASK_ID.h5

julia --project $SLAB_ROOT/recipes/features/scripts/concat.jl \
    -c $compression \
    $out \
    $tmp/[1-9]*h5

