# Real Robot K1 Locomotion Deployment (Booster)

> **See also:** [Inference & Deployment Guide](../../README.md)

This document summarizes the K1 deployment work on `feat/k1`, including runtime fixes and joystick troubleshooting on Jetson.

## Scope

- Robot: Booster `k1_22dof`
- Task: `inference:k1-22dof-loco`
- Model example: `model_24999.onnx`
- Deployment mode: real robot (onboard/offboard), with keyboard or joystick input

---

## 1. Environment Setup

```bash
cd ~/holosoma
source scripts/source_inference_setup.sh
```

Check:

```bash
echo $CONDA_DEFAULT_ENV
python --version
```

Expected:
- `hsinference`
- Python `3.10.x`

---

## 2. Run Command (K1)

### Keyboard mode

```bash
python src/holosoma_inference/holosoma_inference/run_policy.py inference:k1-22dof-loco \
  --task.model-path=/workspace/holosoma/src/holosoma_inference/holosoma_inference/models/loco/k1_22dof/model_24999.onnx \
  --task.interface=usb_eth0
```

### Joystick mode

```bash
python src/holosoma_inference/holosoma_inference/run_policy.py inference:k1-22dof-loco \
  --task.model-path=/workspace/holosoma/src/holosoma_inference/holosoma_inference/models/loco/k1_22dof/model_24999.onnx \
  --task.interface=usb_eth0 \
  --task.use-joystick
```

Notes:
- Correct flag is `--task.use-joystick` (not `--task.joystick`).
- `usb_eth0` must have IPv4 configured.

---

## 3. What Was Fixed During This Deployment

### 3.1 K1 support in Booster SDK path

- Added `k1_22dof` support in:
  - `sdk/command_sender/booster/booster_command_sender.py`
  - `sdk/state_processor/booster/booster_state_processor.py`

### 3.2 Safe startup sequence for Booster

- Restored `prepare command -> ChangeMode(kCustom)` sequence.
- Added validation: `motor_kp/kd` must exist and match motor count before prepare command.
- Prevented accidental prepare with `kp=0` fallback.

### 3.3 Control gain resolution timing

- Reordered policy initialization so ONNX `kp/kd` metadata is resolved before communication components are created.

### 3.4 Startup posture support (locomotion)

- `LocomotionPolicy` now starts in init/default-pose hold mode (`get_ready_state=True`) to avoid tracking a collapsing pose before policy start.

### 3.5 Joystick robustness improvements

- If `--task.use-joystick` is requested but no joystick is available, code now falls back to keyboard with explicit warning.
- `check-botton.py` was upgraded:
  - Prints clear error when no gamepad-like device is found.
  - Prints detected ABS axes and live axis events for stick debugging.
- PS controller button mapping was aligned with observed evdev codes on this Jetson.
- PS right stick yaw mapping was aligned to `ABS_Z` (not `ABS_RX`) based on real logs.

---

## 4. PS Controller Mapping (Observed on This Jetson)

Observed with `check-botton.py`:

- `A`: code `305` (`BTN_EAST`)
- `B`: code `306` (`BTN_C`)
- `X`: code `304` (`BTN_SOUTH`)
- `Y`: code `307` (`BTN_NORTH`)
- `Start`: code `313` (`BTN_TR2`)
- `Select`: code `312` (`BTN_TL2`)
- Right stick yaw: `ABS_Z`

This mapping is now handled in:
- `src/holosoma_inference/holosoma_inference/sdk/command_sender/booster/remote_control_service.py`

---

## 5. Troubleshooting: `lsusb` Sees Controller but Joystick Does Not Work

### Symptom

- `lsusb` shows Sony controller.
- `check-botton.py` prints no gamepad-like device.
- `run_policy --task.use-joystick` shows no response to buttons/sticks.

### Root cause seen in this deployment

- Controller existed at USB/kernel layer, but `/dev/input/event*` node for the gamepad was missing.
- `udevd` was not active, so hotplug node creation did not happen.

### How to diagnose

```bash
lsusb
cat /proc/bus/input/devices
ls -la /dev/input
ls -la /sys/class/input
```

If `/proc/bus/input/devices` shows `Handlers=eventN jsM` for Sony but `/dev/input/eventN` is missing, create nodes manually.

### Manual recovery example

```bash
# Example values from /sys/class/input/*/dev:
# event6 -> 13:70, js0 -> 13:0
sudo mknod /dev/input/event6 c 13 70
sudo mknod /dev/input/js0 c 13 0
sudo chown root:101 /dev/input/event6 /dev/input/js0
sudo chmod 660 /dev/input/event6 /dev/input/js0
```

Then verify:

```bash
python3 -u check-botton.py
```

Expected:
- Device line appears.
- Button and ABS events appear when operating controller.

---

## 6. Safety Checklist Before Walking

1. Use harness/gantry and keep emergency stop ready.
2. Confirm startup logs:
   - ONNX `kp/kd` loaded.
   - DDS discovery completed.
   - No repeated `robot_state_data is None` warnings.
3. In joystick mode, verify button and stick events with `check-botton.py` first.
4. Start policy in place before enabling walking.

---

## 7. Operational Notes

- In locomotion:
  - Start policy: `A` (or keyboard `]`)
  - Stop policy: `B` (or keyboard `o`)
  - Init/default pose: `Y` (or keyboard `i`)
  - Walk/stand toggle: `Start` (or keyboard `=`)
- If controller reconnects, `eventN` can change. Re-check `/proc/bus/input/devices` and `/dev/input`.

