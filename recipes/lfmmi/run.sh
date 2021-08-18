#!/bin/sh -e

# SPDX-License-Identifier: MIT

. $SLAB_ROOT/tools/utils/misc.sh

#######################################################################
# Options

topo_speech_unit=$SLAB_ROOT/tools/lfmmi/conf/hmm_speechunit.toml
topo_nonspeech_unit=$SLAB_ROOT/tools/lfmmi/conf/hmm_nonspeechunit.toml
use_gpu=false
njobs=4
nworkers=4

#######################################################################

show_usage() {
    echo "usage: $(basename $0) [options] <config> <lang-dir> <train-dir> <dev-dir> <init-model> <train-fea> <dev-fea> <out-dir>"
}

show_help() {
    show_usage
    echo ""
    echo "Training a neural network using LF-MMI."
    echo ""
    echo "  --topo-speech-unit     HMM topology file for the speech units (default: $topo_speech_unit)"
    echo "  --topo-nonspeech-unit  HMM topology file for the non-speech units (default: $topo_nonspeech_unit)"
    echo "  --use-gpu              use a GPU to train the model (default: $use_gpu)"
    echo "  --help -h              show this help message"
    echo "  --njobs                number of parallel jobs to comile the FSMs (default: $njobs)"
    echo "  --nworkers             number of parallel workers for data loading (default: $nworkers)"
}

. $SLAB_ROOT/tools/utils/parse_options.sh
if [ $# -ne 8 ]; then
    show_usage 1>&2
    exit 1
fi

config=$1
langdir=$2
traindir=$3
devdir=$4
initmodel=$5
trainfea=$6
devfea=$7
odir=$8

echo "--> Build the HMM components for each basic units."
slab_monophone_mkhmms \
    $topo_speech_unit \
    $topo_nonspeech_unit \
    $langdir/units \
    $odir/hmms.jld2

echo "--> Build the pronunciation fsms."
slab_monophone_mklexicon \
    $langdir/lexicon \
    $odir/lexicon.fsms.jld2

echo "--> Compile the numerator fsms (train set)."
logdir=$odir/logs/align_train
rm -fr $logdir && mkdir -p $odir/logs/align_train
slab_monophone_mkalis \
    --logdir $logdir \
    --njobs $njobs \
    $odir/hmms.jld2 \
    $odir/lexicon.fsms.jld2 \
    $traindir/trans \
    $odir/train_numerator_fsms.jld2

echo "--> Compile the numerator fsms (dev set)."
logdir=$odir/logs/align_dev
rm -fr $logdir && mkdir -p $odir/logs/align_train
slab_monophone_mkalis \
    --logdir $logdir \
    --njobs $njobs \
    $odir/hmms.jld2 \
    $odir/lexicon.fsms.jld2 \
    $devdir/trans \
    $odir/dev_numerator_fsms.jld2

echo "--> Compile the denominator fsm."
slab_monophone_mkploop \
    --start-sil true  \
    --end-sil true \
    $odir/hmms.jld2 \
    $odir/denominator_fsm.jld2


echo "--> Train $initmodel."
$use_gpu && gpu_opt="--use-gpu"
julia -t $((nworkers+1)) --project=@. $SLAB_ROOT/tools/lfmmi/scripts/train.jl \
    $gpu_opt \
    $config \
    $trainfea \
    $devfea \
    $odir/train_numerator_fsms.jld2 \
    $odir/dev_numerator_fsms.jld2 \
    $odir/denominator_fsm.jld2 \
    $initmodel \
    $odir
