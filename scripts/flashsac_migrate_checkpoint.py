#!/usr/bin/env python3
"""Migrate an existing vendored FlashSAC checkpoint directory into holosoma's
single-file ``.ckpt`` format so that ``holosoma.eval_agent`` can ingest it.

Why this exists
---------------
Early FlashSAC training runs (before ``FlashSACAgent.save`` was taught to
produce a single-file ``.ckpt``) only wrote the vendored directory format,
e.g.::

    logs/hv-g1-manager/<timestamp>-g1_29dof_flash_sac_mjwarp_manager-locomotion/
      ├── holosoma_config.yaml
      └── flashsac_step{N}/
          ├── actor.pt
          ├── critic.pt
          ├── target_critic.pt
          ├── temperature.pt
          ├── reward_normalizer.pt  (if normalize_reward was enabled)
          └── agent_state.pt

``eval_agent.py`` calls ``torch.load(checkpoint_path)`` expecting a dict that
contains ``experiment_config`` / ``wandb_run_path`` / ``global_step`` plus the
actual state dicts. This script bundles the directory's per-component files
together with the sibling ``holosoma_config.yaml`` into a single ``.ckpt``
file next to the directory that matches the format ``FlashSACAgent.save``
produces for new runs. After migration, the ``.ckpt`` can be fed straight to
``eval_agent.py --checkpoint <path>.ckpt exp:g1-29dof-flash-sac-mjwarp``.

Usage
-----

    python scripts/flashsac_migrate_checkpoint.py \
        logs/hv-g1-manager/<timestamp>-.../flashsac_step50000

This writes ``flashsac_step50000.ckpt`` next to the ``flashsac_step50000/``
directory in the same run folder. The original directory is left untouched.

Options
-------
    --config <path>     Override the ``holosoma_config.yaml`` path (defaults
                        to ``<run_dir>/holosoma_config.yaml``).
    --output <path>     Override the output ``.ckpt`` path (defaults to
                        ``<run_dir>/<dir_name>.ckpt``).
    --wandb-run-path    Optionally embed a W&B run path in the metadata.
    --global-step       Override the ``global_step`` value. Default: parsed
                        from the directory name (``flashsac_stepN``).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


_COMPONENT_FILES: tuple[str, ...] = (
    "actor.pt",
    "critic.pt",
    "target_critic.pt",
    "temperature.pt",
    "reward_normalizer.pt",
    "agent_state.pt",
)


def _parse_step_from_dir(name: str) -> int:
    match = re.search(r"step(\d+)", name)
    return int(match.group(1)) if match else 0


def _load_experiment_config_dict(yaml_path: Path) -> dict[str, Any]:
    """Load ``holosoma_config.yaml`` as a plain dict suitable for storage in
    the checkpoint metadata. The on-disk format is the output of
    ``ExperimentConfig.to_serializable_dict`` which passes a dataclass→JSON
    round-trip, so the YAML is guaranteed to be a JSON-compatible dict with
    primitive leaves."""

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"holosoma_config.yaml not found at {yaml_path}. "
            f"Pass --config to point at the correct file."
        )
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected a dict at the top level of {yaml_path}, got {type(data)}")
    return data


def migrate(
    checkpoint_dir: Path,
    config_path: Path | None = None,
    output_path: Path | None = None,
    wandb_run_path: str | None = None,
    global_step: int | None = None,
) -> Path:
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"{checkpoint_dir} is not a directory")

    run_dir = checkpoint_dir.parent
    if config_path is None:
        config_path = run_dir / "holosoma_config.yaml"
    if output_path is None:
        output_path = run_dir / f"{checkpoint_dir.name}.ckpt"

    experiment_config = _load_experiment_config_dict(config_path)

    components: dict[str, Any] = {}
    found: list[str] = []
    missing: list[str] = []
    for fname in _COMPONENT_FILES:
        fpath = checkpoint_dir / fname
        if fpath.exists():
            components[fname] = torch.load(str(fpath), map_location="cpu", weights_only=False)
            found.append(fname)
        else:
            missing.append(fname)

    if "actor.pt" not in components or "critic.pt" not in components:
        raise FileNotFoundError(
            f"{checkpoint_dir} is missing required FlashSAC state files. "
            f"Found: {found}. Missing: {missing}"
        )

    step = global_step if global_step is not None else _parse_step_from_dir(checkpoint_dir.name)

    save_dict: dict[str, Any] = {
        "experiment_config": experiment_config,
        "iteration": int(step),
        "global_step": int(step),
        "flashsac_components": components,
    }
    if wandb_run_path:
        save_dict["wandb_run_path"] = wandb_run_path

    torch.save(save_dict, str(output_path))
    print(f"[migrate] wrote {output_path}")
    print(f"[migrate]   source dir : {checkpoint_dir}")
    print(f"[migrate]   config     : {config_path}")
    print(f"[migrate]   components : {found}")
    if missing:
        print(f"[migrate]   (skipped)  : {missing}")
    print(f"[migrate]   global_step: {step}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a FlashSAC checkpoint directory (flashsac_stepN/ containing "
            "actor.pt / critic.pt / target_critic.pt / temperature.pt / "
            "agent_state.pt) into a single-file .ckpt that holosoma's eval_agent "
            "can ingest."
        )
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to the flashsac_stepN/ directory to migrate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional override for holosoma_config.yaml (default: sibling of the dir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the output .ckpt path",
    )
    parser.add_argument("--wandb-run-path", type=str, default=None)
    parser.add_argument("--global-step", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        migrate(
            checkpoint_dir=args.checkpoint_dir,
            config_path=args.config,
            output_path=args.output,
            wandb_run_path=args.wandb_run_path,
            global_step=args.global_step,
        )
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"[migrate] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
