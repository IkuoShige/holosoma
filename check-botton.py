import sys

import evdev


def _is_gamepad_like(device: evdev.InputDevice) -> bool:
    caps = device.capabilities()
    return evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps


def _axis_name(code: int) -> str:
    return str(evdev.ecodes.ABS.get(code, f"ABS_{code}"))


def _norm_axis(value: int, min_value: int, max_value: int) -> float:
    if max_value == min_value:
        return 0.0
    return ((value - min_value) / (max_value - min_value)) * 2.0 - 1.0


def main() -> int:
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

    gamepads = [d for d in devices if _is_gamepad_like(d)]
    if not gamepads:
        print("No gamepad-like input device found (needs both EV_ABS and EV_KEY).")
        print("Visible devices:")
        for d in devices:
            caps = d.capabilities()
            has_key = evdev.ecodes.EV_KEY in caps
            has_abs = evdev.ecodes.EV_ABS in caps
            print(f"  {d.path}: {d.name} (EV_KEY={has_key}, EV_ABS={has_abs})")
        return 1

    # Pick the first matching device to keep behavior simple.
    d = gamepads[0]
    caps = d.capabilities()
    abs_entries = caps.get(evdev.ecodes.EV_ABS, [])
    abs_map = {code: absinfo for code, absinfo in abs_entries}

    print(f"Device: {d.name} ({d.path})")
    print("Press buttons/sticks. Ctrl+C to quit.")
    print("Detected ABS axes:")
    for code, absinfo in abs_entries:
        print(
            f"  code={code:3d} name={_axis_name(code):12s} "
            f"range=[{absinfo.min}, {absinfo.max}] center={absinfo.value}"
        )

    if evdev.ecodes.ABS_RX not in abs_map or evdev.ecodes.ABS_RY not in abs_map:
        print("WARNING: ABS_RX/ABS_RY not fully present. Right stick may be on different axis codes.")
        print("         Check live ABS event output below and update runtime mapping accordingly.")

    for event in d.read_loop():
        if event.type == evdev.ecodes.EV_KEY and event.value == 1:
            name = evdev.ecodes.BTN.get(event.code, evdev.ecodes.KEY.get(event.code, "?"))
            print(f"  Button code: {event.code} name: {name}")
        elif event.type == evdev.ecodes.EV_ABS:
            absinfo = abs_map.get(event.code)
            if absinfo is None:
                print(f"  Axis code: {event.code} name: {_axis_name(event.code)} value={event.value}")
            else:
                norm = _norm_axis(event.value, absinfo.min, absinfo.max)
                print(
                    f"  Axis code: {event.code} name: {_axis_name(event.code)} "
                    f"value={event.value:5d} norm={norm:+.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
