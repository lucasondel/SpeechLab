# SPDX-License-Identifier: MIT

# Set of utility functions use by various scripts.

error() {
    echo $1 1>&2
    exit 1
}

assert_not_missing() {
    if [ ! -f $1 ]; then
        echo "Missing file: $1." 1>&2
        exit 1
    fi
}

