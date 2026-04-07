#!/usr/bin/env bash
# Gate B (mjwarp variant) smoke test for the holosoma-native FlashSAC adapter.
#
# Same agent / bridge / experiment shape as run_flashsac_holosoma_smoke.sh, but
# uses the GPU-accelerated MuJoCo Warp backend instead of IsaacSim. Runs against
# ``LeggedRobotLocomotionManager`` for 5 outer interaction steps.
#
# Requires the ``hsmujoco`` conda env (Python 3.10, torch 2.10, mujoco 3.6,
# mujoco_warp). hydra-core and gymnasium must be installed in that env (see
# docs/flashsac_port.md).
#
# Tunables (env vars):
#   FLASHSAC_NUM_ENVS   number of parallel envs (default 64)
#   FLASHSAC_SEED       RNG seed (default 0)
#   FLASHSAC_NUM_ITERS  outer interaction-step count (default 5)

set -eo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

# source_mujoco_setup.sh appends to LD_LIBRARY_PATH unconditionally; pre-init it
# so that sourcing under ``set -u`` does not abort with "unbound variable".
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${SCRIPT_DIR}/source_mujoco_setup.sh"
set -u

NUM_ENVS=${FLASHSAC_NUM_ENVS:-64}
SEED=${FLASHSAC_SEED:-0}
NUM_ITERS=${FLASHSAC_NUM_ITERS:-5}

echo "[smoke] num_envs=${NUM_ENVS}, seed=${SEED}, num_iterations=${NUM_ITERS} (mjwarp)"

cd "${PROJECT_ROOT}"

python src/holosoma/holosoma/train_agent.py exp:g1-29dof-flash-sac-mjwarp \
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
