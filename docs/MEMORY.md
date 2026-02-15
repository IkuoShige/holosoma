# Holosoma Project Memory

## Environment Setup
- Use `source scripts/source_mujoco_setup.sh` (or isaacgym/isaacsim variant) before running Python
- Ruff may not be installed in conda env; install with `pip install ruff` if needed
- ONNX export requires `onnxscript` package (not always installed)

## Code Patterns
- Agent directories (ppo/, fast_sac/, fpo/) have NO `__init__.py` files
- Config types use `pydantic.dataclasses` with `frozen=True`
- Config values provide default instances and register in DEFAULTS dict
- Experiment presets use `dataclasses.replace()` to customize algo configs
- `AlgoConfig` / `AlgoInitConfig` are Union types — extend when adding new algos
- `_update_algo_step` expects loss dict keys: value_loss, surrogate_loss, entropy_loss, kl_mean (popped), extras handled generically
- Import sorting: ruff expects `from holosoma.*` before `from loguru/torch`

## ONNX Export Pitfalls
- `_OnnxMotionPolicyExporter` / `_extract_actor_model_and_input_dim` はPPO/FastSACのactor構造を前提にwrapperを分解する。カスタムwrapperを使う場合は `export()` ごとオーバーライドが必要
- forward内で `torch.arange`/`torch.full`/`torch.zeros` を使うとONNX constant foldingでCPU/CUDA不整合が起きる。対策: `register_buffer`, `tensor.new_full()`, `tensor.new_zeros()`
- export前にモデル全体を `.cpu()` に移すのが最も確実

## FPO Implementation (2025-02-15)
- See [fpo-work-log.md](fpo-work-log.md) for full work log and training commands
- Files: `agents/fpo/fpo_agent.py`, `agents/modules/flow_policy.py`
- Config: `FPOConfig`, `FPOModuleDictConfig`, `FPOAlgoConfig` in `config_types/algo.py`
- Inherits PPO, overrides: _setup_models_and_optimizer, _setup_storage, _rollout_step, _compute_ppo_loss, _post_epoch_logging, export, actor_onnx_wrapper, get_inference_policy
- Interpolation: x_t = t*eps + (1-t)*action (t=0:clean, t=1:noise)
- ODE integration: from t=1→0, step = x + dt*v_theta
