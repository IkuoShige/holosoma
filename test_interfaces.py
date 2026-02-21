"""Test which network interface receives B1LowState messages."""
import time
import netifaces as ni
from booster_robotics_sdk import ChannelFactory, B1LowStateSubscriber

# List all interfaces with IPv4
print("Available interfaces:")
for iface in ni.interfaces():
    addrs = ni.ifaddresses(iface).get(ni.AF_INET, [])
    if addrs:
        print(f"  {iface}: {addrs[0]['addr']}")
    else:
        print(f"  {iface}: (no IPv4)")

# Test each interface with IPv4
for iface in ["usb_eth0", "enP8p1s0", "enP9p1s0"]:
    addrs = ni.ifaddresses(iface).get(ni.AF_INET, [])
    if not addrs:
        print(f"\n=== {iface}: skipped (no IPv4) ===")
        continue
    ip = addrs[0]["addr"]
    print(f"\n=== Testing {iface} (IP: {ip}) ===")

    received = []

    def make_handler(name):
        def handler(msg):
            received.append(time.time())
            if len(received) <= 2:
                print(f"  [{name}] got state msg #{len(received)}, rpy={msg.imu_state.rpy}")
        return handler

    ChannelFactory.Instance().Init(0, ip)
    sub = B1LowStateSubscriber(make_handler(iface))
    sub.InitChannel()
    time.sleep(3)
    status = "OK" if received else "FAIL"
    print(f"  [{status}] {iface}: {len(received)} messages in 3s")
