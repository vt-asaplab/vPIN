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
    echo "  -l       Run LeNet Model"
    echo "  -b       Run baby-step-giant-step algorithm to precompute the table"
    echo "  -d       Run server and client with specified arguments"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

# Set the project directory to the current working directory.
PROJECT_DIR=$(pwd)

# Function to run server and client for the specified CNN network version.
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
    python3 "$PROJECT_DIR/src/cnn_networks/Server.py" "$version" "$port" &
    SERVER_PID=$!
    sleep 2
    python3 "$PROJECT_DIR/src/cnn_networks/Client.py" "$port"
    wait $SERVER_PID
}

# Function to run server and client with the specified filter and image sizes.
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
    python3 "$PROJECT_DIR/src/convolution/Server.py" "$version" "$port" &
    SERVER_PID=$!
    sleep 2
    python3 "$PROJECT_DIR/src/convolution/Client.py" "$size" "$port"
    wait $SERVER_PID
}

# Function to run the LeNet model server and client.
run_server_and_client3() {
    local port=$1

    echo "Running LeNet Model (server.py and client.py) on port $port..."
    python3 "$PROJECT_DIR/src/LeNet/Server.py" "$port" &
    SERVER_PID=$!
    sleep 2
    python3 "$PROJECT_DIR/src/LeNet/Client.py" "$port"
    wait $SERVER_PID
}

# Main script execution based on provided command-line arguments.
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
                run_server_and_client 1 8085
                ;;
            -B)
                run_server_and_client 2 8095
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
        run_server_and_client3 8315
        ;;
    -b)
        echo "Running baby-step-giant-step.py..."
        python3 "$PROJECT_DIR/src/Pre_computed_table/baby-step-giant-step.py"
        ;;
    -d)
        if [ $# -ne 3 ]; then
            usage
        fi
        run_server_and_client2 $2 $3 8158
        ;;
    *)
        usage
        ;;
esac

echo "Script execution completed."
