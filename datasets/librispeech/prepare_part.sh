#!/bin/sh -e

# SPDX-License-Identifier: MIT

# This script is adapted from:
# https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/local/data_prep.sh

. $SLAB_ROOT/utils/misc.sh

show_usage() {
    echo "usage: $(basename $0) <part-dir> <out-dir>"
}

show_help() {
    show_usage
    echo ""
    echo "Prepare a part of the LibriSpeech ASR corpus."
    echo ""
    echo "  -h, --help      show this help message"
}

. $SLAB_ROOT/utils/parse_options.sh
if [ $# -ne 2 ]; then
    show_usage 1>&2
    exit 1
fi

partdir=$1
odir=$2
mkdir -p $odir

spk_file=$partdir/../SPEAKERS.TXT

assert_is_installed flac
assert_not_missing $spk_file

wav_scp=$odir/wav.scp; rm -f $wav_scp
trans=$odir/trans.wrd; rm -f $trans
utt2spk=$odir/utt2spk; rm -f $utt2spk
uttids=$odir/uttids; rm -f $uttids

for reader_dir in $(find -L $partdir -mindepth 1 -maxdepth 1 -type d | sort); do
    reader=$(basename $reader_dir)
    for chapter_dir in $(find -L $reader_dir -mindepth 1 -maxdepth 1 -type d | sort); do
        chapter=$(basename $chapter_dir)
        find -L $chapter_dir/ -iname "*.flac" | \
                    sort | \
                    xargs -I% basename % .flac | \
                    awk -v "dir=$chapter_dir" '{printf "lbi-%s\tflac -c -d -s %s/%s.flac |\n", $0, dir, $0}' \
                    >> $odir/wav.scp

        chapter_trans=$chapter_dir/${reader}-${chapter}.trans.txt
        sed -e 's/^/lbi\-/' $chapter_trans | sed -e 's/ /\t/' >> $odir/trans.wrd

        awk -v "reader=$reader" -v "chapter=$chapter" '{printf "lbi-%s\tlbi-%s-%s\n", $1, reader, chapter}' \
            <$chapter_trans >> $odir/utt2spk
    done
done

cut -f1 $odir/wav.scp > $odir/uttids

