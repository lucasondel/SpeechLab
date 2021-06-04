#!/bin/sh -e

show_usage() {
    echo "usage: $(basename $0) <host> <port> <file>"
}

show_help() {
    show_usage
    echo ""
    echo "Serve <file> for HTTP GET connection coming from <host>:<port>."
    echo ""
    echo "  -h              show this help message"
}

if [ $# -ne 3 ]; then
    show_usage 1>&2
    exit 1
fi


host=$1
port=$2
file=$3

http_response() {
    file=$1

    echo HTTP/1.1 200 OK
    echo Content-type: application/json
    echo ""

    cat $file | grep "^{"
}



while true; do
    http_response $file | nc -l $host $port
done

