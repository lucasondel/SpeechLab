#!/bin/sh -e

# SPDX-License-Identifier: MIT

# This script is an adapted version of:
#   https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/parse_options.sh

. $SLAB_ROOT/utils/misc.sh

cmdname=$(basename $0)

while :; do

    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --*)
            name=$(echo "$1" | sed s/^--// | sed s/-/_/g)
            if eval '[ -z "${'$name'+xxx}" ]'; then
                error "Unkown option: \"$1\".  Try \"$cmdname -h\" for more information."
            fi

            oldval="$(eval echo \$$name)"
            if [ "$oldval" = "true" ] || [ "$oldval" = "false" ]; then
                boolean=true
            else
                boolean=false
            fi

            eval $name=\"$2\"

            if $boolean && [ ! "$2" = true  ] && [ ! "$2" = "false" ]; then
                error "Expected \"true\" or \"false\": $1 $2."
            fi

            shift 2
            ;;
        -*) error "Short options are not supported: $1 $2";;
        *) break;;
    esac
done
