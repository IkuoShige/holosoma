"""Diagnostic: test which init step breaks the B1LowStateSubscriber callback."""
import time
import netifaces as ni
from booster_robotics_sdk import (
    ChannelFactory,
    B1LowStateSubscriber,
    B1LowCmdPublisher,
    B1LocoClient,
    LowCmd,
    LowCmdType,
    MotorCmd,
    RobotMode,
)

INTERFACE = "enP8p1s0"
ip = ni.ifaddresses(INTERFACE)[ni.AF_INET][0]["addr"]
print(f"Using IP: {ip}")

received = []


def handler(msg):
    received.append(time.time())
    if len(received) <= 3:
        print(f"[CB] received state message #{len(received)}, rpy={msg.imu_state.rpy}")


def wait_and_check(label, seconds=3):
    before = len(received)
    time.sleep(seconds)
    after = len(received)
    got = after - before
    status = "OK" if got > 0 else "FAIL"
    print(f"  [{status}] {label}: received {got} messages in {seconds}s")
    return got > 0


# === Test 1: Standalone (baseline) ===
print("\n=== Test 1: ChannelFactory.Init + Subscriber only ===")
ChannelFactory.Instance().Init(0, ip)
sub = B1LowStateSubscriber(handler)
sub.InitChannel()
wait_and_check("Subscriber alone")

# === Test 2: Add B1LowCmdPublisher ===
print("\n=== Test 2: + B1LowCmdPublisher.InitChannel() ===")
pub = B1LowCmdPublisher()
pub.InitChannel()
wait_and_check("After B1LowCmdPublisher")

# === Test 3: Add B1LocoClient.Init() ===
print("\n=== Test 3: + B1LocoClient.Init() ===")
client = B1LocoClient()
client.Init()
wait_and_check("After B1LocoClient.Init()")

# === Test 4: ChangeMode(kCustom) ===
print("\n=== Test 4: + ChangeMode(kCustom) ===")
client.ChangeMode(RobotMode.kCustom)
wait_and_check("After ChangeMode(kCustom)")

# === Test 5: Re-init ChannelFactory ===
print("\n=== Test 5: Re-init ChannelFactory ===")
ChannelFactory.Instance().Init(0, ip)
wait_and_check("After ChannelFactory re-init")

# === Test 6: Create new subscriber after all the above ===
print("\n=== Test 6: New subscriber after everything ===")
received2 = []


def handler2(msg):
    received2.append(time.time())
    if len(received2) <= 3:
        print(f"[CB2] received state message #{len(received2)}")


sub2 = B1LowStateSubscriber(handler2)
sub2.InitChannel()
before = len(received2)
time.sleep(3)
after = len(received2)
got = after - before
status = "OK" if got > 0 else "FAIL"
print(f"  [{status}] New subscriber: received {got} messages in 3s")

print("\nDone. Summary:")
print(f"  Original subscriber total: {len(received)} messages")
print(f"  New subscriber total: {len(received2)} messages")
