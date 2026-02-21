import evdev
devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
for d in devices:
    caps = d.capabilities()
    if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
        print(f'Device: {d.name}')
        print('Press each button (A, B, X, Y, Start, Select). Ctrl+C to quit.')
        for event in d.read_loop():
            if event.type == evdev.ecodes.EV_KEY and event.value == 1:
                name = evdev.ecodes.BTN.get(event.code, evdev.ecodes.KEY.get(event.code, '?'))
                print(f'  Button code: {event.code} name: {name}')
        break
