# SPDX-License-Identifier: MIT

# Set of utility functions use by various scripts.

assert_not_missing() {
    if [ ! -f $1 ]; then
        echo "missing file: $1" 1>&2
        exit 1
    fi
}
