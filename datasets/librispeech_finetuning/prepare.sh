#!/bin/sh -e
# SPDX-License-Identifier: MIT

show_usage() {
    echo "usage: $(basename $0) [options] <dst-dir>"
}

show_help() {
    show_usage
    echo ""
    echo "Prepare the LibriSpeech_finetune ASR corpus. See: https://github.com/facebookresearch/libri-light/tree/master/data_preparation#2-get-the-limited-supervision-train-data"
    echo "for details"
    echo ""
    echo "  -h              show this help message"
}

while :; do
    case "$1" in
        -h) show_help; exit 0;;
        --) shift; break;;
        -*)
            echo "Unknown option: $1. Try '$0 -h' for more informations."  1>&2
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
    wget https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
    tar -xf librispeech_finetuning.tgz
    cd $curdir
    date > $localdir/corpus/.done
else
    echo "Data already downloaded. Skipping."
fi

prepare_subset() {
    sdir=$1
    subsetdir=$2
    for reader_dir in $(find -L $subsetdir -mindepth 1 -maxdepth 1 -type d | sort); do
        reader=$(basename $reader_dir)
        for chapter_dir in $(find -L $reader_dir -mindepth 1 -maxdepth 1 -type d | sort); do
            chapter=$(basename $chapter_dir)
            find -L $chapter_dir/ -iname "*.flac" | \
                sort | \
                xargs -I% basename % .flac | \
                awk -v "dir=$chapter_dir" '{printf "%s\tflac -c -d -s %s/%s.flac |\n", $0, dir, $0}' \
                >> $sdir/wav.scp;

                chapter_trans=$chapter_dir/${reader}-${chapter}.trans.txt
                sed -e 's/ /\t/' $chapter_trans >> $sdir/trans.wrd

                awk -v "reader=$reader" -v "chapter=$chapter" '{printf "%s\t%s-%s\n", $1, reader, chapter}' \
                    <$chapter_trans >> $sdir/utt2spk || exit 1
        done
    done
    cut -f1 $sdir/wav.scp > $sdir/uttids
}

# 10 minutes split.
sdir=$odir/10min
if [ ! -f $sdir/.done ]; then
    echo "Preparing 10min split..."
    mkdir -p $sdir
    rm -f $sdir/wav.scp $sdir/trans.wrd $sdir/utt2spk

    for subset in clean other; do
        subsetdir=$localdir/corpus/librispeech_finetuning/1h/0/$subset
        prepare_subset $sdir $subsetdir
    done
    date > $sdir/.done
else
    echo "10min split already prepared. Skipping."
fi

# 1 hour split.
sdir=$odir/1h
if [ ! -f $sdir/.done ]; then
    echo "Preparing 1h split..."
    mkdir -p $sdir
    rm -f $sdir/wav.scp $sdir/trans.wrd $sdir/utt2spk

    for subset in clean other; do
        for n in $(seq 0 5); do
            subsetdir=$localdir/corpus/librispeech_finetuning/1h/$n/$subset
            prepare_subset $sdir $subsetdir
        done
    done
    date > $sdir/.done
else
    echo "1h split already prepared. Skipping."
fi

# 10 hour split.
sdir=$odir/10h
if [ ! -f $sdir/.done ]; then
    echo "Preparing 10h split..."
    mkdir -p $sdir
    rm -f $sdir/wav.scp $sdir/trans.wrd $sdir/utt2spk

    for subset in clean other; do
        for n in $(seq 0 5); do
            subsetdir=$localdir/corpus/librispeech_finetuning/1h/$n/$subset
            prepare_subset $sdir $subsetdir
        done
    done

    for subset in clean other; do
        subsetdir=$localdir/corpus/librispeech_finetuning/9h/$subset
        prepare_subset $sdir $subsetdir
    done
    date > $sdir/.done
else
    echo "10h split already prepared. Skipping."
fi

