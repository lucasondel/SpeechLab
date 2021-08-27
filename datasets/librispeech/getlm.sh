#!/bin/sh -e

# SPDX-License-Identifier: MIT

. $SLAB_ROOT/utils/misc.sh

show_usage() {
    echo "usage: $(basename $0) <url> <out-dir>"
}

show_help() {
    show_usage
    echo ""
    echo "Download the language models of the LibriSpeech corpus."
    echo ""
    echo "  -h, --help      show this help message"
}


. $SLAB_ROOT/utils/parse_options.sh
if [ $# -ne 2 ]; then
    show_usage 1>&2
    exit 1
fi

url=$1
odir=$2
mkdir -p $odir

for lmfile in 3-gram.arpa.gz 3-gram.pruned.1e-7.arpa.gz \
              3-gram.pruned.3e-7.arpa.gz 4-gram.arpa.gz g2p-model-5 \
              librispeech-lm-corpus.tgz librispeech-vocab.txt \
              librispeech-lexicon.txt; do
    if [ ! -f $odir/$lmfile ]; then
        wget -O $odir/$lmfile $url/$lmfile
    else
        echo "$lmfile already downloaded in $odir/$lmfile."
    fi
done
