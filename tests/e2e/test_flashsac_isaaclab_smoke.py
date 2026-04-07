"""End-to-end smoke test for the vendored FlashSAC port (Gate A).

Runs the full vendored Hydra-driven training pipeline against the IsaacLab
stock G1 locomotion task for exactly 5 interaction steps inside a subprocess
that activates the ``hssim`` conda env. This is a regression guard for the
"完全移植" port: if any vendored module breaks, this test fails before any
other agent does.

Marked ``@pytest.mark.isaacsim @pytest.mark.slow`` because it requires:

- the ``hssim`` conda env (Python 3.11, IsaacSim 5.1.0, IsaacLab 2.3.0)
- a CUDA-capable GPU with ~6 GB free
- ~60 seconds wall-clock for IsaacSim cold start

Run manually with::

    pytest -m "isaacsim and slow" tests/e2e/test_flashsac_isaaclab_smoke.py -v -s
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "run_flashsac_isaaclab_smoke.sh"
RUNS_ROOT = REPO_ROOT / "runs" / "smoke" / "g1_5step"
TIMEOUT_S = 900


@pytest.mark.isaacsim
@pytest.mark.slow
def test_flashsac_g1_5step_smoke(tmp_path: Path) -> None:
    """The vendored FlashSAC train.py runs G1 for 5 interaction steps and writes a non-empty TensorBoard event file."""

    assert SMOKE_SCRIPT.exists(), f"smoke script missing: {SMOKE_SCRIPT}"

    # Capture the set of pre-existing run directories so we can identify the
    # one this invocation creates and ignore stale ones from previous runs.
    pre_existing = {p.name for p in RUNS_ROOT.glob("Isaac-Velocity-Flat-G1-v0_seed0_*")} if RUNS_ROOT.exists() else set()

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT)],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )

    if result.returncode != 0:
        # Surface the last few KB of stderr/stdout in the failure message so
        # debugging the test does not require digging through CI logs.
        tail = (result.stderr or "")[-3000:] + "\n---STDOUT---\n" + (result.stdout or "")[-1000:]
        pytest.fail(f"smoke script exit code {result.returncode}; tail:\n{tail}")

    assert "5/5" in result.stdout or "5/5" in result.stderr, "tqdm did not reach 5/5"

    assert RUNS_ROOT.exists(), f"runs root not created: {RUNS_ROOT}"

    new_runs = [
        p for p in RUNS_ROOT.glob("Isaac-Velocity-Flat-G1-v0_seed0_*")
        if p.name not in pre_existing
    ]
    assert new_runs, f"smoke run did not create a new TensorBoard directory under {RUNS_ROOT}"

    new_run_dir = max(new_runs, key=lambda p: p.stat().st_mtime)
    event_files = list(new_run_dir.glob("events.out.tfevents.*"))
    assert event_files, f"no TensorBoard event file in {new_run_dir}"
    assert all(f.stat().st_size > 0 for f in event_files), "TensorBoard event file is empty"

    # Verify the event file actually contains FlashSAC training scalars.
    ea = EventAccumulator(str(new_run_dir))
    ea.Reload()
    scalar_tags = set(ea.Tags().get("scalars", []))
    required = {"critic/loss", "actor/loss", "temperature/value"}
    missing = required - scalar_tags
    assert not missing, f"smoke event file missing FlashSAC scalars: {missing}; got {sorted(scalar_tags)}"

    critic_loss_events = ea.Scalars("critic/loss")
    assert len(critic_loss_events) >= 1, "no critic/loss events recorded — update() never ran"
