#!/usr/bin/env bash
# Gate A smoke test for the vendored FlashSAC port.
#
# Runs the vendored Hydra-driven train.py against the IsaacLab stock G1
# locomotion task (Isaac-Velocity-Flat-G1-v0) for exactly 5 interaction steps,
# with eval/recording/checkpointing disabled, no torch.compile, no AMP, and a
# tiny replay buffer warm-up. The goal is to prove the entire vendored code
# path (configs, agent, networks, buffer, IsaacLab env wrapper, training loop)
# works end-to-end inside the hssim conda env, NOT to actually train anything.
#
# Tunables (can be overridden via env vars):
#   FLASHSAC_NUM_ENVS   number of parallel IsaacLab envs (default 64)
#   FLASHSAC_SEED       RNG seed (default 0)

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

source "${SCRIPT_DIR}/source_isaacsim_setup.sh"

NUM_ENVS=${FLASHSAC_NUM_ENVS:-64}
SEED=${FLASHSAC_SEED:-0}
NUM_ENV_STEPS=$(( NUM_ENVS * 5 ))  # 5 interaction steps total

echo "[smoke] num_envs=${NUM_ENVS}, seed=${SEED}, num_env_steps=${NUM_ENV_STEPS}"

cd "${PROJECT_ROOT}"

python -m holosoma._vendored.flash_rl.train \
  --config_name flashSAC_base \
  --overrides "env=isaaclab" \
  --overrides "env.env_name=Isaac-Velocity-Flat-G1-v0" \
  --overrides "num_env_steps=${NUM_ENV_STEPS}" \
  --overrides "num_train_envs=${NUM_ENVS}" \
  --overrides "num_eval_envs=null" \
  --overrides "num_record_envs=null" \
  --overrides "num_eval_episodes=0" \
  --overrides "num_record_episodes=0" \
  --overrides "evaluation_per_interaction_step=0" \
  --overrides "recording_per_interaction_step=0" \
  --overrides "metrics_per_interaction_step=0" \
  --overrides "logging_per_interaction_step=1" \
  --overrides "save_checkpoint_per_interaction_step=0" \
  --overrides "save_buffer_per_interaction_step=null" \
  --overrides "agent.use_compile=false" \
  --overrides "agent.use_amp=false" \
  --overrides "agent.buffer_min_length=${NUM_ENVS}" \
  --overrides "agent.sample_batch_size=${NUM_ENVS}" \
  --overrides "agent.buffer_max_length=4096" \
  --overrides "agent.normalize_reward=false" \
  --overrides "logger_type=tensorboard" \
  --overrides "group_name=smoke" \
  --overrides "exp_name=g1_5step" \
  --overrides "seed=${SEED}"
