#!/bin/bash
# Source this script to activate the mjlab conda environment.
#
# Usage:
#   source scripts/source_mjlab_setup.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsmjlab}

source ${SCRIPT_DIR}/source_common.sh

if [[ ! -d $CONDA_ROOT/envs/$CONDA_ENV_NAME ]]; then
    echo "ERROR: mjlab conda env not found. Run: bash scripts/setup_mjlab.sh"
    return 1 2>/dev/null || exit 1
fi

source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME
echo "mjlab environment activated ($CONDA_ENV_NAME)"
