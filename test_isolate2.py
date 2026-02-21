"""Test: does waiting for state before B1LocoClient fix ChangeMode?"""
import time
import sys
import netifaces as ni
from booster_robotics_sdk import (
    ChannelFactory,
    B1LowStateSubscriber,
    B1LowCmdPublisher,
    B1LocoClient,
    RobotMode,
)

INTERFACE = "usb_eth0"
ip = ni.ifaddresses(INTERFACE)[ni.AF_INET][0]["addr"]

test = sys.argv[1] if len(sys.argv) > 1 else "A"
print(f"Running test {test} (IP: {ip})")

ChannelFactory.Instance().Init(0, ip)

got_state = [False]
def handler(msg):
    got_state[0] = True

sub = B1LowStateSubscriber(handler)
sub.InitChannel()

if test == "A":
    # Wait for state BEFORE client init (like test_custom_mode3.py)
    print("Test A: Wait for state -> client.Init -> ChangeMode")
    print("Waiting for state...")
    for i in range(50):
        if got_state[0]:
            print(f"  Got state after {i*0.1:.1f}s")
            break
        time.sleep(0.1)
    if not got_state[0]:
        print("  No state after 5s!")

elif test == "B":
    # Just wait 2 seconds (no state check)
    print("Test B: sleep(2) -> client.Init -> ChangeMode")
    time.sleep(2)

elif test == "C":
    # No wait at all
    print("Test C: No wait -> client.Init -> ChangeMode")

pub = B1LowCmdPublisher()
pub.InitChannel()
client = B1LocoClient()
client.Init()
result = client.ChangeMode(RobotMode.kCustom)
print(f"ChangeMode result: {result}")
print("Is the light BLUE? (waiting 5s)")
time.sleep(5)
