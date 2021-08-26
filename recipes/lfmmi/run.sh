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

echo "================================================================"
echo "Training"
echo "================================================================"
mkdir -p $expdir/train
slab_lfmmi_train \
    --topo-nonspeech-unit conf/topo_unit.toml \
    --topo-speech-unit conf/topo_unit.toml \
    --use-gpu true \
    --njobs 10 \
    --nworkers 4 \
    conf/train_config.toml \
    $datasetdir/lang \
    $datasetdir/train \
    $datasetdir/dev \
    models/tdnn.jld2 \
    $feadir/train/$featype.h5 \
    $feadir/dev/$featype.h5 \
    $expdir/train | tee $expdir/train/log

