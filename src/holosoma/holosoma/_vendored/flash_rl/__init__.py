"""Vendored copy of FlashSAC's ``flash_rl`` package.

This package is a verbatim mirror of the upstream FlashSAC ``flash_rl/`` tree.
The only allowed mechanical edits are:

1. ``from flash_rl...`` and ``import flash_rl...`` rewritten to
   ``holosoma._vendored.flash_rl...``.
2. ``train.py`` refactored into ``build_cfg / run / main(argv=None)`` so it
   can be reused in-process by holosoma's adapter layer
   (``holosoma.agents.flash_sac``).
3. ``configs/`` lives at ``holosoma/_vendored/flash_rl/configs/``; the
   vendored ``train.py`` resolves the path via
   ``Path(__file__).parent / "configs"`` so it works regardless of cwd.
4. ``flash_rl/envs/isaaclab.py`` adds ``import isaaclab_tasks`` so that
   IsaacLab task IDs (e.g. ``Isaac-Velocity-Flat-G1-v0``) are registered with
   ``gymnasium`` before ``parse_env_cfg`` is called.

The package is exempt from holosoma's lint rules; see ``pyproject.toml``.
Treat ``holosoma._vendored.*`` as an internal compatibility layer, not a
stable public API.

This module also installs FlashSAC's ``${eval: ...}`` OmegaConf resolver in
an idempotent way so that repeated Hydra recomposes (e.g. inside pytest) do
not raise.
"""

from omegaconf import OmegaConf

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))  # noqa: S307,PGH001
