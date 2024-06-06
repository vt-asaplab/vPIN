#!/bin/bash

set -e

usage() {
    echo "Usage: $0 [-a | -c -A|-B|-C|-D|-E | -l | -b | -d <3|5|7> <32|64|128|256>]"
    echo "  -a       Check accuracy"
    echo "  -c -A    Run CNN network A"
    echo "  -c -B    Run CNN network B"
    echo "  -c -C    Run CNN network C"
    echo "  -c -D    Run CNN network D"
    echo "  -c -E    Run CNN network E"
    echo "  -l       Run lenet/code.py"
    echo "  -b       Run baby-step-giant-step algorithm to precompute the table"
    echo "  -d       Run server and client with specified arguments"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

PROJECT_DIR=$(pwd)

run_server_and_client() {
    local version=$1
    local port=$2
    local network_label

    case $version in
        1)
            network_label="A"
            ;;
        2)
            network_label="B"
            ;;
        3)
            network_label="C"
            ;;
        4)
            network_label="D"
            ;;
        5)
            network_label="E"
            ;;
        *)
            echo "Invalid version number: $version"
            exit 1
            ;;
    esac

    echo "Running server.py and client.py for CNN network $network_label on port $port..."
    python3 "$PROJECT_DIR/src/cnn_networks/server.py" "$version" "$port" &
    SERVER_PID=$!
    sleep 2
    python3 "$PROJECT_DIR/src/cnn_networks/client.py" "$port"
    wait $SERVER_PID
}

run_server_and_client2() {
    local version=$1
    local size=$2
    local port=$3

    if [[ ! "$version" =~ ^(3|5|7)$ ]]; then
        echo "Invalid version number: $version. Allowed values are 3, 5, or 7."
        exit 1
    fi

    if [[ ! "$size" =~ ^(32|64|128|256)$ ]]; then
        echo "Invalid size: $size. Allowed values are 32, 64, 128, or 256."
        exit 1
    fi

    echo "Running server.py with filter size $version and client.py with image size $size on port $port..."
    python3 "$PROJECT_DIR/src/convolution/server.py" "$version" "$port" &
    SERVER_PID=$!
    sleep 2
    python3 "$PROJECT_DIR/src/convolution/client.py" "$size" "$port"
    wait $SERVER_PID
}

case $1 in
    -a)
        echo "Running accuracy/train_test_lenet5.py..."
        python3 "$PROJECT_DIR/src/accuracy/train_test_lenet5.py"
        ;;
    -c)
        if [ $# -ne 2 ]; then
            usage
        fi
        case $2 in
            -A)
                run_server_and_client 1 8087
                ;;
            -B)
                run_server_and_client 2 8092
                ;;
            -C)
                run_server_and_client 3 8088
                ;;
            -D)
                run_server_and_client 4 8089
                ;;
            -E)
                run_server_and_client 5 8090
                ;;
            *)
                usage
                ;;
        esac
        ;;
    -l)
        echo "Running lenet/code.py..."
        python3 "$PROJECT_DIR/lenet/code.py"
        ;;
    -b)
        echo "Running baby-step-giant-step.py..."
        python3 "$PROJECT_DIR/src/cnn_networks/Pre_computed_table/baby-step-giant-step.py"
        ;;
    -d)
        if [ $# -ne 3 ]; then
            usage
        fi
        run_server_and_client2 $2 $3 8138
        ;;
    *)
        usage
        ;;
esac

echo "Script execution completed."
