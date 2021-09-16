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
graphsdir=$expdir/graphs

#######################################################################

mkdir -p $expdir

echo "================================================================"
echo "Features extraction"
echo "================================================================"

for dir in $traindir $devdir $testdir; do
    setname=$(basename $dir)
    logdir=$feadir/$setname/logs/${featype} && mkdir -p $logdir
    slab_features_extract \
        --logdir $logdir \
        --njobs 10 \
        $feaconfig \
        $dir/wav.scp \
        $feadir/$setname/${featype}.h5
done

echo "================================================================"
echo "Graph (numerator/denominator) preparation"
echo "================================================================"

mkdir -p $graphsdir

echo "--> Build the HMM components for each unit."
slab_hmm_mkhmms \
    conf/topo_unit.toml \
    $langdir/units \
    $graphsdir/hmms.jld2

echo "--> Build the pronunciation FSMs."
slab_hmm_mklexicon \
    $langdir/lexicon \
    $graphsdir/lexicon_fsms.jld2

for dir in $traindir $devdir; do
    dname=$(basename $dir)
    echo "--> Compile the numerator FSMs ($dname set)."
    logdir=$graphsdir/logs/make-alis-${dname} && mkdir -p $logdir
    slab_hmm_mkalis \
        --edge-silprob 0.8 \
        --between-silprob 0.1 \
        --logdir $logdir \
        --njobs 10 \
        $graphsdir/hmms.jld2 \
        $graphsdir/lexicon_fsms.jld2 \
        $dir/trans.wrd \
        $graphsdir/${dname}_numerator_fsms.jld2
done

echo "--> Build the denominator FSM."
slab_lfmmi_mkdenfsm \
    --between-silprob 0.1 \
    --edge-silprob 0.8 \
    $ngram_order \
    $graphsdir/hmms.jld2 \
    $langdir/lexicon \
    $traindir/trans.wrd \
    $graphsdir/den_fsm.jld2

echo "================================================================"
echo "Training"
echo "================================================================"

mkdir -p $expdir/train
mkdir -p $expdir/train/checkpoints

initmodel=$expdir/train/init.jld2
finalmodel=$expdir/train/final.jld2

echo "--> Create the initial model using $modelfile."
slab_lfmmi_mkmodel \
    $modelfile \
    $modelconfig \
    $feadir/$(basename $traindir)/${featype}.h5 \
    $graphsdir/hmms.jld2 \
    $initmodel

echo "--> Starting training..."
logfile=$expdir/train/training.log
echo "Training started at $(date)." > $logfile
slab_lfmmi_train \
    --checkpoint-dir $expdir/train/checkpoints \
    $ckpt_opts \
    --use-gpu true \
    $modelfile \
    $trainconfig \
    $modelconfig \
    $feadir/$(basename $traindir)/${featype}.h5 \
    $feadir/$(basename $devdir)/${featype}.h5 \
    $graphsdir/$(basename $traindir)_numerator_fsms.jld2 \
    $graphsdir/$(basename $devdir)_numerator_fsms.jld2 \
    $graphsdir/den_fsm.jld2 \
    $initmodel | tee -a $logfile
echo "Finished training at $(date)." >> $logfile

# Output of the training.
finalmodel=$expdir/train/checkpoints/best.jld2

echo "================================================================"
echo "Generating output"
echo "================================================================"

mkdir -p $expdir/output

echo "--> Dumping model's output."
slab_lfmmi_dump \
    --use-gpu true \
    --batch-size 200 \
    $modelfile \
    $modelconfig \
    $finalmodel \
    $feadir/$(basename $testdir)/${featype}.h5 \
    $expdir/output/$(basename $testdir).h5

echo "--> Converting to Kaldi format."
python h5_to_kaldi.py \
    $expdir/output/$(basename $testdir).h5 \
    $expdir/output/$(basename $testdir).ark
