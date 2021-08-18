#!/bin/sh -e

# SPDX-License-Identifier: MIT

. $SLAB_ROOT/tools/utils/misc.sh
scriptdir=$SLAB_ROOT/datasets/mini_librispeech

MINI_LIBRISPEECH_URL=https://www.openslr.org/resources/31

show_usage() {
    echo "usage: $(basename $0) <out-dir>"
}

show_help() {
    show_usage
    echo ""
    echo "Prepare the Mini LibriSpeech ASR corpus."
    echo "See: $MINI_LIBRISPEECH_URL for details."
    echo ""
    echo "  -h, --help      show this help message"
}

. $SLAB_ROOT/tools/utils/parse_options.sh
if [ $# -ne 1 ]; then
    show_usage 1>&2
    exit 1
fi

odir=$1
curdir=$(pwd)
localdir=$odir/.local
mkdir -p $odir/.local

assert_is_installed flac

mkdir -p $localdir
for part in dev-clean-2 train-clean-5; do
    $scriptdir/getdata.sh $MINI_LIBRISPEECH_URL $part $localdir
done

$scriptdir/prepare_part.sh $localdir/LibriSpeech/dev-clean-2 $odir/dev
$scriptdir/prepare_part.sh $localdir/LibriSpeech/train-clean-5 $odir/train

# Use the development set as test set.
cdir=$PWD
cd $odir
ln -s ./dev "test"
cd $cdir

