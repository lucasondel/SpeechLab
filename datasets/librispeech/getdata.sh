#!/bin/sh -e

# SPDX-License-Identifier: MIT

. $SLAB_ROOT/utils/misc.sh

show_usage() {
    echo "usage: $(basename $0) <url> <part> <out-dir>"
}

show_help() {
    show_usage
    echo ""
    echo "Download and extract part of the LibriSpeech ASR corpus."
    echo ""
    echo "  -h, --help      show this help message"
}


. $SLAB_ROOT/utils/parse_options.sh
if [ $# -ne 3 ]; then
    show_usage 1>&2
    exit 1
fi

url=$1
part=$2
odir=$3
mkdir -p $odir

if [ ! -f "$odir/${part}.tar.gz" ]; then
    cdir=$(pwd)
    cd $odir/
    wget $url/${part}.tar.gz
    cd $cdir
else
    echo "$part already downloaded in $odir/${part}.tar.gz."
fi

if [ ! -f "$odir/LibriSpeech/$part/.done" ]; then
    tar -C $odir -xvf $odir/${part}.tar.gz
    date > $odir/LibriSpeech/$part/.done
else
    echo "Archive already extracted in $odir/$part."
fi
