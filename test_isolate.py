"""Isolate what breaks ChangeMode."""
import time
import sys
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

INTERFACE = "usb_eth0"
NUM_MOTORS = 22
ip = ni.ifaddresses(INTERFACE)[ni.AF_INET][0]["addr"]

test = sys.argv[1] if len(sys.argv) > 1 else "A"
print(f"Running test {test} (IP: {ip})")

ChannelFactory.Instance().Init(0, ip)
sub = B1LowStateSubscriber(lambda msg: None)
sub.InitChannel()

if test == "A":
    # Same as test_custom_mode3.py (WORKS): create+init one by one
    print("Test A: pub.Init -> client.Init -> ChangeMode (no prepare cmd)")
    pub = B1LowCmdPublisher()
    pub.InitChannel()
    client = B1LocoClient()
    client.Init()
    result = client.ChangeMode(RobotMode.kCustom)

elif test == "B":
    # Full app order: create both, then init both
    print("Test B: create both -> init both -> ChangeMode (no prepare cmd)")
    pub = B1LowCmdPublisher()
    client = B1LocoClient()
    pub.InitChannel()
    client.Init()
    result = client.ChangeMode(RobotMode.kCustom)

elif test == "C":
    # Full app order + prepare cmd
    print("Test C: create both -> init both -> send prepare cmd -> ChangeMode")
    pub = B1LowCmdPublisher()
    client = B1LocoClient()
    pub.InitChannel()
    client.Init()
    # Send prepare cmd with kp=0
    cmd = LowCmd()
    cmd.cmd_type = LowCmdType.SERIAL
    mc = [MotorCmd() for _ in range(NUM_MOTORS)]
    cmd.motor_cmd = mc
    pub.Write(cmd)
    result = client.ChangeMode(RobotMode.kCustom)

elif test == "D":
    # Same order as working test but NO state wait
    print("Test D: pub.Init -> client.Init -> ChangeMode (create+init sequential, no wait)")
    pub = B1LowCmdPublisher()
    pub.InitChannel()
    client = B1LocoClient()
    client.Init()
    result = client.ChangeMode(RobotMode.kCustom)

print(f"ChangeMode result: {result}")
print("Is the light BLUE? (waiting 5s then exit)")
time.sleep(5)
