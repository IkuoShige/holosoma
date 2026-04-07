#!/usr/bin/env bash
# Gate B smoke test for the holosoma-native FlashSAC adapter.
#
# Runs holosoma's standard ``train_agent.py`` with the new
# ``exp:g1-29dof-flash-sac`` subcommand. This exercises the Layer 2 adapter
# (FlashSACAgent + FlashSACGymBridge) against the actual
# ``LeggedRobotLocomotionManager`` task on the IsaacSim backend.
#
# Tunables (env vars):
#   FLASHSAC_NUM_ENVS   number of parallel envs (default 64)
#   FLASHSAC_SEED       RNG seed (default 0)
#   FLASHSAC_NUM_ITERS  outer interaction-step count (default 5)

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

source "${SCRIPT_DIR}/source_isaacsim_setup.sh"

NUM_ENVS=${FLASHSAC_NUM_ENVS:-64}
SEED=${FLASHSAC_SEED:-0}
NUM_ITERS=${FLASHSAC_NUM_ITERS:-5}

echo "[smoke] num_envs=${NUM_ENVS}, seed=${SEED}, num_iterations=${NUM_ITERS}"

cd "${PROJECT_ROOT}"

python src/holosoma/holosoma/train_agent.py exp:g1-29dof-flash-sac \
  --algo.config.num-learning-iterations="${NUM_ITERS}" \
  --algo.config.buffer-min-length="${NUM_ENVS}" \
  --algo.config.sample-batch-size="${NUM_ENVS}" \
  --algo.config.buffer-max-length=4096 \
  --algo.config.use-compile=False \
  --algo.config.use-amp=False \
  --algo.config.normalize-reward=False \
  --algo.config.updates-per-interaction-step=1.0 \
  --algo.config.logging-interval=1 \
  --algo.config.save-interval=0 \
  --training.num-envs="${NUM_ENVS}" \
  --training.seed="${SEED}" \
  --training.headless=True
