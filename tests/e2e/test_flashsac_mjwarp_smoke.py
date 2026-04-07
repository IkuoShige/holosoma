"""End-to-end smoke test for the holosoma FlashSAC adapter on MuJoCo Warp.

Mirrors ``test_flashsac_holosoma_smoke.py`` but uses the ``hsmujoco`` conda env
and the ``g1_29dof_flash_sac_mjwarp`` experiment, which targets the GPU
``mujoco_warp`` backend instead of IsaacSim. Same FlashSAC algorithm, same
``LeggedRobotLocomotionManager`` env, different physics backend.

Marked ``@pytest.mark.mujoco @pytest.mark.slow``. Requires the ``hsmujoco``
env (Python 3.10, torch 2.10, mujoco 3.6, mujoco_warp) with ``hydra-core`` and
``gymnasium`` installed (see ``docs/flashsac_port.md``).

Run manually with::

    pytest -m "mujoco and slow" tests/e2e/test_flashsac_mjwarp_smoke.py -v -s
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "run_flashsac_mjwarp_smoke.sh"
LOGS_ROOT = REPO_ROOT / "logs" / "hv-g1-manager"
TIMEOUT_S = 900


@pytest.mark.mujoco
@pytest.mark.slow
def test_flashsac_mjwarp_g1_5step_smoke() -> None:
    """End-to-end FlashSAC adapter run on G1 + MuJoCo Warp."""

    assert SMOKE_SCRIPT.exists(), f"smoke script missing: {SMOKE_SCRIPT}"

    pre_existing = (
        {p.name for p in LOGS_ROOT.glob("*flash_sac_mjwarp*")}
        if LOGS_ROOT.exists()
        else set()
    )

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT)],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )

    if result.returncode != 0:
        tail = (result.stderr or "")[-3000:] + "\n---STDOUT---\n" + (result.stdout or "")[-1000:]
        pytest.fail(f"smoke script exit code {result.returncode}; tail:\n{tail}")

    assert "5/5" in result.stdout or "5/5" in result.stderr, "tqdm did not reach 5/5"

    new_runs = [p for p in LOGS_ROOT.glob("*flash_sac_mjwarp*") if p.name not in pre_existing]
    assert new_runs, f"smoke run did not create a holosoma run dir under {LOGS_ROOT}"

    new_run_dir = max(new_runs, key=lambda p: p.stat().st_mtime)
    event_files = list(new_run_dir.glob("events.out.tfevents.*"))
    assert event_files, f"no TensorBoard event file in {new_run_dir}"
    assert all(f.stat().st_size > 0 for f in event_files), "TensorBoard event file is empty"

    config_path = new_run_dir / "holosoma_config.yaml"
    assert config_path.exists(), f"holosoma_config.yaml missing in {new_run_dir}"

    checkpoint_dir = new_run_dir / "flashsac_step5"
    assert checkpoint_dir.exists(), f"flashsac checkpoint dir missing: {checkpoint_dir}"
    expected = {"actor.pt", "critic.pt", "target_critic.pt", "temperature.pt", "agent_state.pt"}
    actual = {p.name for p in checkpoint_dir.iterdir()}
    missing = expected - actual
    assert not missing, f"checkpoint dir missing files: {missing} (got {actual})"

    ea = EventAccumulator(str(new_run_dir))
    ea.Reload()
    scalar_tags = set(ea.Tags().get("scalars", []))
    required = {"flashsac/critic/loss", "flashsac/actor/loss", "flashsac/temperature/value"}
    missing_tags = required - scalar_tags
    assert not missing_tags, (
        f"holosoma TensorBoard log missing FlashSAC scalars: {missing_tags}; got {sorted(scalar_tags)}"
    )
    critic_events = ea.Scalars("flashsac/critic/loss")
    assert len(critic_events) >= 1, "no critic/loss events recorded — adapter did not run update()"
