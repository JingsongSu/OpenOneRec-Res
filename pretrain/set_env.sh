#!/bin/bash

# Check if current shell is bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script must be run with bash. Please use 'bash script.bash' to run it." >&2
    exit 1
fi

# Get current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# Check if .env file exists
if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: ${ENV_FILE} not found" >&2
    exit 1
fi

# Load environment variables
set -a  # Automatically export all variables
source "${ENV_FILE}"
set +a  # Disable automatic export

# Print loaded environment variables
echo "Loaded environment variables from ${ENV_FILE}:"
cat "${ENV_FILE}"

# Install system dependencies
PIP_CMD='pip'
PROXY="http://oversea-squid1.jp.txyun:11080"
HOSTFILE="/etc/mpi/hostfile"

# Install numactl on all nodes
mpirun --allow-run-as-root \
    --hostfile "${HOSTFILE}" \
    -x http_proxy="${PROXY}" \
    -x https_proxy="${PROXY}" \
    --pernode \
    bash -c "apt-get install -y numactl"

# Install Python dependencies on all nodes
mpirun --allow-run-as-root \
    --hostfile "${HOSTFILE}" \
    --pernode \
    bash -c "${PIP_CMD} install transformers==4.53 && \
             ${PIP_CMD} install easydict && \
             ${PIP_CMD} install torchao==0.10 && \
             ${PIP_CMD} install sortedcontainers"
