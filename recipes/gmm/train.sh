#!/bin/sh -e

cmd=$0
root=$(dirname $(dirname $(dirname $(realpath $cmd))))
. $root/env.sh
. $root/utils/utils.sh

#######################################################################
# Defaults

nutts_init=10000
use_gpu=no
double_precision=no
checkpoint_rate=10

#######################################################################

show_usage() {
    echo "usage: $cmd [options] <dataset> <fea-name> <config>"
}

show_help() {
    show_usage
    echo ""
    echo "Train a GMM on <dataset>."
    echo ""
    echo "  -c C            save a checkpoint every C updates"
    echo "  -d              use double precision for computations"
    echo "                  (default is single precision)"
    echo "  -g              use a GPU for training the model"
    echo "  -h              show this help message"
    echo "  -u U            number of utterances to use to calculate"
    echo "                  the data statistics for initalization"
    echo "                  (default: $nutts_init)"
}

while :; do
    case "$1" in
        -h) show_help; exit 0;;
        -c) checkpoint_rate=$2; shift;;
        -d) double_precision=yes;;
        -g) use_gpu=yes;;
        -u) nutts_init=$2; shift;;
        -*)
            echo "unknown option: $1. Try '$0 -h' for more informations."  1>&2
            echo "" 1>&2
            show_usage 1>&2
            exit 1;;
        *) break;;
    esac
    shift
done

if [ $# -ne 3 ]; then
    show_usage 1>&2
    exit 1
fi

datasetname=$1
feaname=$2
config=$3

datasetdir=$SLAB_DATASETS/$datasetname
features=$SLAB_FEATURES/$datasetname/$feaname.h5

# The model name corresponds to the name of the configuration file
# without the extension.
bname=$(basename $config)
modelname=${bname%%.*}

odir=$SLAB_MODELS/$datasetname/$modelname
mkdir -p $odir

echo "The model and other intermediate results will be stored in $odir."

uttids=$datasetdir/uttids

assert_not_missing $uttids
assert_not_missing $features

stats=$odir/stats.h5
if [ ! -f $stats ]; then
    shuf < $uttids | head -n $nutts_init | \
        julia --project scripts/feastats.jl $features $stats
else
    echo "Statistics are already estimated ($stats)."
fi

[ $use_gpu = "yes" ] && gpu_opt="-g"
[ $double_precision = "no" ] && precision_opt="-S"

checkpoint_name=$(find $odir -name "[0-9]*jld2" -exec basename {} \; | \
    sort -t '.' -k 1 -g | tail -1)
if [ ! -z $checkpoint_name ]; then
    num=$(echo $checkpoint_name | cut -d '.' -f1)
    start_update=$((num + 1))
else
    start_update=1
fi
checkpoint=$odir/$checkpoint_name

final=$odir/final.jld2
if [ ! -f $final ]; then
    cat $uttids | julia --project scripts/gmm.jl \
        $precision_opt $gpu_opt \
        --checkpoint-rate $checkpoint_rate \
        --start-from $checkpoint \
        --start-update $start_update \
        $config \
        $stats \
        $features \
        $odir | tee $odir/log
else
    echo "The model is already trained ($final)."
fi
