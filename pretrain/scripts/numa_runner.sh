#!/bin/bash

# Get local NUMA node count
num_numa=$(numactl -H | grep "node [0-9] cpus" | wc -l)
if [ "$num_numa" -lt 1 ]; then
  num_numa=1
fi

# Default to NUMA 0
numa_id=0

echo "Bind to NUMA node $numa_id"

# Bind memory and CPU to NUMA node 0 when running command
numactl --membind=$numa_id --cpunodebind=$numa_id "$@"