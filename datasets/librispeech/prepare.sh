#!/bin/sh -e
# SPDX-License-Identifier: MIT

show_usage() {
    echo "usage: $(basename $0) [options] <dst-dir>"
}

show_help() {
    show_usage
    echo ""
    echo "Prepare the LibriSpeech ASR corpus. See: http://www.openslr.org/12"
    echo "for details"
    echo ""
    echo "  -h              show this help message"
}

while :; do
    case "$1" in
        -h) show_help; exit 0;;
        --) shift; break;;
        -*)
            echo "unknown option: $1. Try '$0 -h' for more informations."  1>&2
            echo "" 1>&2
            show_usage 1>&2
            exit 1;;
        *) break;;
    esac
    shift
done

if [ $# -ne 1 ]; then
    show_usage 1>&2
    exit 1
fi

odir=$1
curdir=$(pwd)
localdir=$odir/.local
mkdir -p $odir/.local

[ -x "$(command -v flac)" ] || (echo "flac is not installed" && exit 1)

if [ ! -f "$localdir/corpus/.done" ]; then
    mkdir -p $localdir/corpus
    cd $localdir/corpus
    wget https://www.openslr.org/resources/12/dev-clean.tar.gz
    wget https://www.openslr.org/resources/12/dev-other.tar.gz
    wget https://www.openslr.org/resources/12/test-clean.tar.gz
    wget https://www.openslr.org/resources/12/test-other.tar.gz
    wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
    wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
    wget https://www.openslr.org/resources/12/train-other-500.tar.gz

    for archive in *tar.gz; do
        echo "extracting archive: $archive..."
        tar -xf $archive
    done

    cd $curdir
    date > $localdir/corpus/.done
fi

subsets="dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500"

for subset in $subsets; do

    subsetdir=$localdir/corpus/LibriSpeech/$subset
    sdir=$odir/$subset
    if [ ! -f $sdir/.done ]; then
        echo preparing subset: $subset


        mkdir -p $sdir

        rm -f $sdir/wav.scp
        rm -f $sdir/trans.wrd
        rm -f $sdir/utt2spk

        for reader_dir in $(find -L $subsetdir -mindepth 1 -maxdepth 1 -type d | sort); do
            reader=$(basename $reader_dir)
            for chapter_dir in $(find -L $reader_dir -mindepth 1 -maxdepth 1 -type d | sort); do
                chapter=$(basename $chapter_dir)
                find -L $chapter_dir/ -iname "*.flac" | \
                    sort | \
                    xargs -I% basename % .flac | \
                    awk -v "dir=$chapter_dir" '{printf "lbi-%s\tflac -c -d -s %s/%s.flac |\n", $0, dir, $0}' \
                    >> $sdir/wav.scp || exit 1;

                chapter_trans=$chapter_dir/${reader}-${chapter}.trans.txt
                sed -e 's/^/lbi\-/' $chapter_trans | sed -e 's/ /\t/' >> $sdir/trans.wrd

                awk -v "reader=$reader" -v "chapter=$chapter" '{printf "lbi-%s\tlbi-%s-%s\n", $1, reader, chapter}' \
                    <$chapter_trans >> $sdir/utt2spk || exit 1
            done
        done

        cut -f1 $sdir/wav.scp > $sdir/uttids

        date > $sdir/.done
    else
        echo "$subset is already prepared"
    fi
done

if [ ! -f $odir/train-all/.done ]; then
    echo preparing subset: train-all
    mkdir -p $odir/train-all

    cat $odir/train-clean-100/wav.scp > $odir/train-all/wav.scp
    cat $odir/train-clean-360/wav.scp >> $odir/train-all/wav.scp
    cat $odir/train-other-500/wav.scp >> $odir/train-all/wav.scp

    cat $odir/train-clean-100/uttids > $odir/train-all/uttids
    cat $odir/train-clean-360/uttids >> $odir/train-all/uttids
    cat $odir/train-other-500/uttids >> $odir/train-all/uttids

    cat $odir/train-clean-100/utt2spk > $odir/train-all/utt2spk
    cat $odir/train-clean-360/utt2spk >> $odir/train-all/utt2spk
    cat $odir/train-other-500/utt2spk >> $odir/train-all/utt2spk

    date > $odir/train-all/.done
else
    echo "train-all is already prepared"
fi
