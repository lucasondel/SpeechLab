#!/bin/sh -e

# SPDX-License-Identifier: MIT

. $SLAB_ROOT/tools/utils/misc.sh

cmdname=$(basename $0)

while :; do

    #if [ -z "$2" ]; then
    #    echo "empty argument to $1 option" 1>&2
    #    exit 1
    #fi

    case "$1" in
        --help|-h)
            show_help()
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
        *) break;;
    esac
done
