#!/bin/bash
# Setup script for mjlab simulator backend.
#
# Creates a conda environment with mjlab and holosoma installed.
# Requires: /workspace/mjlab/ to exist with mjlab source.
#
# Usage:
#   bash scripts/setup_mjlab.sh

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
MJLAB_DIR="/workspace/mjlab"

CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsmjlab}
echo "conda environment name is set to: $CONDA_ENV_NAME"

source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_$CONDA_ENV_NAME

mkdir -p $WORKSPACE_DIR

MJLAB_REPO="https://github.com/mujocolab/mjlab.git"
MJLAB_TAG="v1.1.1"

if [ ! -d "$MJLAB_DIR" ]; then
    echo "mjlab not found at $MJLAB_DIR, cloning ${MJLAB_REPO} (tag: ${MJLAB_TAG})..."
    git clone --branch "$MJLAB_TAG" --depth 1 "$MJLAB_REPO" "$MJLAB_DIR"
fi

if [[ ! -f $SENTINEL_FILE ]]; then
  OS_NAME="$(uname -s)"
  ARCH_NAME="$(uname -m)"

  # Install miniconda (reuse existing logic)
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT

    if [[ "$OS_NAME" == "Linux" ]]; then
      MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OS_NAME" == "Darwin" ]]; then
      if [[ "$ARCH_NAME" == "arm64" ]]; then
        MINICONDA_INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
      else
        MINICONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
      fi
    else
      echo "Unsupported OS: $OS_NAME"
      exit 1
    fi

    curl "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" -o "$CONDA_ROOT/miniconda.sh"
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $CONDA_ENV_NAME python=3.10 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME

  if [[ "$OS_NAME" == "Linux" ]]; then
    conda install -c conda-forge -y libstdcxx-ng
  fi
  conda install -c conda-forge -y ffmpeg

  pip install --upgrade pip

  # Install mjlab (editable)
  echo "Installing mjlab from $MJLAB_DIR..."
  pip install -e "$MJLAB_DIR"

  # Install holosoma
  echo "Installing holosoma..."
  if [[ "$OS_NAME" == "Linux" ]]; then
    pip install -e "$ROOT_DIR/src/holosoma[unitree, booster]"
  elif [[ "$OS_NAME" == "Darwin" ]]; then
    pip install -e "$ROOT_DIR/src/holosoma[unitree]"
  else
    echo "Unsupported OS: $OS_NAME"
    exit 1
  fi

  # Validate
  echo "Validating mjlab installation..."
  python -c "import mjlab; print('mjlab imported successfully')"
  python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
  python -c "from holosoma.simulator.mjlab import MjLab; print('MjLab simulator imported successfully')"

  touch $SENTINEL_FILE

  echo ""
  echo "=========================================="
  echo "mjlab environment setup completed!"
  echo "=========================================="
  echo ""
  echo "Activate with: source scripts/source_mjlab_setup.sh"
  echo "=========================================="
fi

echo ""
echo "mjlab environment ready."
echo "Use 'source scripts/source_mjlab_setup.sh' to activate."
