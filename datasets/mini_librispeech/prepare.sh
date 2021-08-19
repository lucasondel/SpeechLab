#!/bin/sh -e

# SPDX-License-Identifier: MIT

. $SLAB_ROOT/tools/utils/misc.sh
scriptdir=$SLAB_ROOT/datasets/mini_librispeech

MINI_LIBRISPEECH_URL=https://www.openslr.org/resources/31
MINI_LIBRISPEECH_LM_URL=https://www.openslr.org/resources/11

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
mkdir -p $localdir

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
rm -f test
ln -s ./dev "test"
cd $cdir

# Download the Language Models.
$scriptdir/getlm.sh $MINI_LIBRISPEECH_LM_URL $localdir/lms

# Prepare the 'lang' directory.
langdir=$odir/lang
mkdir -p $langdir

python $scriptdir/filter_lexicon.py $localdir/lms/librispeech-lexicon.txt \
    > $langdir/lexicon

echo "SIL\tnonspeech-unit" > $langdir/units
echo "SPN\tnonspeech-unit" >> $langdir/units
while read line; do
    echo "$line\tspeech-unit"
done <$scriptdir/cmudict_phones >> $langdir/units

