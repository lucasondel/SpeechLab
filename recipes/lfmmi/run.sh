#!/bin/sh -e

#######################################################################
# Settings

## Input dataset ##
dataset=mini_librispeech
datasetdir=~/Datasets/$dataset
langdir=$datasetdir/lang
traindir=$datasetdir/train
devdir=$datasetdir/dev
testdir=$datasetdir/test

## Features configuration ##
feadir=~/Features/$dataset
featype=mfcc_hires_16kHz
feaconfig=conf/${featype}.toml

## Denominator graph ##
ngram_order=3  # order of the phonotactic language model

## Model ##
modelfile=$PWD/models/tdnn.jl # must be an absolute path
modelconfig=conf/tdnn.toml
trainconfig=conf/train.toml

## Experiment output ##
#expdir=exp/$dataset/${ngram_order}gram
expdir=exp/$dataset #/${ngram_order}gram
logdir=$expdir/logs

#######################################################################

mkdir -p $expdir

echo "================================================================"
echo "Features extraction"
echo "================================================================"

for dir in $traindir $devdir $testdir; do
    setname=$(basename $dir)
    mkdir -p $logdir/fea-extract-$setname
    slab_features_extract \
        --logdir $logdir/fea-extract-$setname \
        --njobs 10 \
        $feaconfig \
        $dir/wav.scp \
        $feadir/$setname/${featype}.h5
done

echo "================================================================"
echo "Graph (numerator/denominator) preparation"
echo "================================================================"

echo "--> Build the HMM components for each unit."
slab_hmm_mkhmms \
    conf/topo_unit.toml \
    $langdir/units \
    $expdir/hmms.jld2

echo "--> Build the pronunciation FSMs."
slab_hmm_mklexicon \
    $langdir/lexicon \
    $expdir/lexicon_fsms.jld2

for dir in $traindir $devdir; do
    dname=$(basename $dir)
    echo "--> Compile the numerator FSMs ($dname set)."
    mkdir -p $logdir/make-alis-$dname
    slab_hmm_mkalis \
        --logdir $logdir/make-alis-$dname \
        --njobs 10 \
        $expdir/hmms.jld2 \
        $expdir/lexicon_fsms.jld2 \
        $dir/trans.wrd \
        $expdir/${dname}_numerator_fsms.jld2
done

echo "--> Build the denominator FSM."
slab_lfmmi_mkdenfsm \
    --between-silprob 0.1 \
    --edge-silprob 0.8 \
    $ngram_order \
    $expdir/hmms.jld2 \
    $langdir/lexicon \
    $traindir/trans.wrd \
    $expdir/den_fsm.jld2

echo "================================================================"
echo "Training"
echo "================================================================"

mkdir -p $expdir/train

echo "--> Create the initial model using $modelfile."
initmodel=$expdir/train/init.jld2
if [ ! -f $initmodel ]; then
    julia --project scripts/mkmodel.jl \
        $modelfile \
        $modelconfig \
        $feadir/$(basename $traindir)/${featype}.h5 \
        $expdir/hmms.jld2 \
        $initmodel
else
    echo "Initial model already created in $initmodel"
fi

# Check if there is a checkpoint.
last_ckpt=$expdir/train/checkpoints/last.jld2
[ -f $last_ckpt ] && ckpt_opts="--from-checkpoint $last_ckpt"

echo "--> Starting training..."
finalmodel=$expdir/train/final.jld2
if [ ! -f $finalmodel ]; then
    slab_lfmmi_train \
        --checkpoint-dir $expdir/train/checkpoints \
        $ckpt_opts \
        --use-gpu true \
        --njobs 10 \
        --nworkers 4 \
        $modelfile \
        $trainconfig \
        $modelconfig \
        $feadir/$(basename $traindir)/${featype}.h5 \
        $feadir/$(basename $devdir)/${featype}.h5 \
        $expdir/$(basename $traindir)_numerator_fsms.jld2 \
        $expdir/$(basename $devdir)_numerator_fsms.jld2 \
        $expdir/den_fsm.jld2 \
        $initmodel \
        $finalmodel
else
    echo "Model already trained in $initmodel"
fi

