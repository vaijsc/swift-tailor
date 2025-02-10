#!/bin/bash

n_processes=$1

# Kill direct descendants of this script
cleanup() {
    echo "Terminating all subprocesses..."
    pkill -P $$  # Kills all child processes of this script
    exit 1
}

# Trap Ctrl + C and script termination
trap cleanup SIGINT SIGTERM

# Generate UVs
mkdir -p logs
for i in $(seq 0 $((n_processes-1)));
do
    python scripts/generate_uv.py --subset-id $i --num-subsets $n_processes >> "logs/${i}.txt" 2>&1 &
done

wait

echo "All processes completed."