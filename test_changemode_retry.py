"""Test: retry ChangeMode until robot enters custom mode."""
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

INTERFACE = "usb_eth0"
NUM_MOTORS = 22
ip = ni.ifaddresses(INTERFACE)[ni.AF_INET][0]["addr"]
print(f"Using IP: {ip}")

ChannelFactory.Instance().Init(0, ip)

# Subscriber
got_state = [False]
def handler(msg):
    got_state[0] = True

sub = B1LowStateSubscriber(handler)
sub.InitChannel()

print("Waiting for state...")
for i in range(50):
    if got_state[0]:
        print(f"  Got state after {i*0.1:.1f}s")
        break
    time.sleep(0.1)

# Publisher + Client
pub = B1LowCmdPublisher()
pub.InitChannel()
client = B1LocoClient()
client.Init()

# Try ChangeMode with retries
for attempt in range(5):
    print(f"\nAttempt {attempt+1}: ChangeMode(kCustom)...")
    result = client.ChangeMode(RobotMode.kCustom)
    print(f"  result: {result}")

    # Send some commands to verify
    print("  Sending commands for 2s...")
    for i in range(100):
        cmd = LowCmd()
        cmd.cmd_type = LowCmdType.SERIAL
        mc = [MotorCmd() for _ in range(NUM_MOTORS)]
        for j in range(NUM_MOTORS):
            mc[j].q = 0.0
            mc[j].kp = 50.0
            mc[j].kd = 2.0
        cmd.motor_cmd = mc
        pub.Write(cmd)
        time.sleep(0.02)

    print("  Is the light BLUE? Check and press Enter (or type 'y' if blue):")
    ans = input("  > ").strip().lower()
    if ans == 'y':
        print("  Custom mode confirmed!")
        break
    print("  Not in custom mode, retrying...")
    time.sleep(1)

print("\nSending hold commands for 5s...")
end = time.time() + 5
while time.time() < end:
    cmd = LowCmd()
    cmd.cmd_type = LowCmdType.SERIAL
    mc = [MotorCmd() for _ in range(NUM_MOTORS)]
    for j in range(NUM_MOTORS):
        mc[j].q = 0.0
        mc[j].kp = 50.0
        mc[j].kd = 2.0
    cmd.motor_cmd = mc
    pub.Write(cmd)
    time.sleep(0.02)

# Clean exit test
print("\nTesting clean exit modes...")
for attr in dir(RobotMode):
    if not attr.startswith('_'):
        print(f"  RobotMode.{attr} = {getattr(RobotMode, attr)}")

print("Done.")
