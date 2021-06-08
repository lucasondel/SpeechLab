# SPDX-License-Identifier: MIT

# Set of utility functions use by various scripts.

assert_not_missing() {
    if [ ! -f $1 ]; then
        echo "Missing file: $1." 1>&2
        exit 1
    fi
}

assert_is_dataset() {
    dataset=$1
    if [ ! -f $SLAB_DATASETS/$dataset/wav.scp ]; then
        echo "Unknown dataset: $dataset. Check that this dataset exist in $SLAB_DATASETS." 1>&2
        echo "Note that a dataset directory should contain at least a 'wav.scp' file." 1>&2
        exit 1
    fi
}
