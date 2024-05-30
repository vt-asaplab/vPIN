#!/bin/bash

set -e

usage() {
    echo "Usage: $0 [-a | -c | -l]"
    echo "  -a    check accuracy"
    echo "  -c    Run CNN/code.py"
    echo "  -l    Run lenet/code.py"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

PROJECT_DIR=$(pwd)

case $1 in
    -a)
        echo "Running accuracy/test.py..."
        python3 "$PROJECT_DIR/src/accuracy/train_test_lenet5.py"
        ;;
    -c)
        echo "Running CNN/code.py..."
        python3 "$PROJECT_DIR/CNN/code.py"
        ;;
    -l)
        echo "Running lenet/code.py..."
        python3 "$PROJECT_DIR/lenet/code.py"
        ;;
    *)
        usage
        ;;
esac

echo "Script execution completed."
