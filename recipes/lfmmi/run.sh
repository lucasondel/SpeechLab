#!/bin/sh -e

# SPDX-License-Identifier: MIT

. $SLAB_ROOT/tools/utils/misc.sh

#######################################################################
# Settings

## Input dataset ##
dataset=mini_librispeech
datasetdir=~/Datasets/$dataset
trainset=$datasetdir/train
devset=$datasetdir/dev
testset=$datasetdir/test

## Experiment output ##
expdir=exp
logdir=exp/logs

## Features configuration ##
feadir=~/Features/$dataset
featype=mfcc_hires_16kHz
feaconfig=conf/${featype}.toml

#######################################################################

mkdir -p $expdir $logdir

echo "================================================================"
echo "Extracting features"
echo "================================================================"
for subset in $trainset $devset $testset; do
    mkdir -p $logdir/fea-extract
    slab_features_extract \
        --logdir $logdir/fea-extract \
        --njobs 10 \
        $feaconfig \
        $subset/wav.scp \
        $feadir/$(basename $subset)/${featype}.h5
done
exit 0

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
