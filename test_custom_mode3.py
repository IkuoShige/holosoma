"""Test: verify robot responds to motor commands in custom mode."""
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

# Capture state once
positions = [0.0] * NUM_MOTORS
got_state = [False]


def state_handler(msg):
    if not got_state[0]:
        for i in range(NUM_MOTORS):
            positions[i] = msg.motor_state_serial[i].q
        got_state[0] = True


sub = B1LowStateSubscriber(state_handler)
sub.InitChannel()

print("Waiting for state...")
for i in range(50):
    if got_state[0]:
        break
    time.sleep(0.1)

if got_state[0]:
    print(f"Got state! First 6 joints: {[f'{q:.3f}' for q in positions[:6]]}")
else:
    print("No state - using defaults")
    positions = [0.0, 0.0, 0.2, -1.35, 0.0, -0.5, 0.2, 1.35, 0.0, 0.5,
                 -0.2, 0.0, 0.0, 0.4, -0.25, 0.0, -0.2, 0.0, 0.0, 0.4, -0.25, 0.0]

pub = B1LowCmdPublisher()
pub.InitChannel()
client = B1LocoClient()
client.Init()

result = client.ChangeMode(RobotMode.kCustom)
print(f"ChangeMode result: {result}")
time.sleep(0.5)

# Pre-build command template
target_positions = list(positions)  # Copy

print(f"\nSending hold commands for 10s (kp=50, kd=2)...")
print(">>> Push the robot gently - should resist if custom mode works <<<")

end = time.time() + 10
count = 0
while time.time() < end:
    cmd = LowCmd()
    cmd.cmd_type = LowCmdType.SERIAL
    motor_cmds = [MotorCmd() for _ in range(NUM_MOTORS)]
    for i in range(NUM_MOTORS):
        motor_cmds[i].q = target_positions[i]
        motor_cmds[i].dq = 0.0
        motor_cmds[i].tau = 0.0
        motor_cmds[i].kp = 50.0
        motor_cmds[i].kd = 2.0
    cmd.motor_cmd = motor_cmds
    pub.Write(cmd)
    count += 1
    if count % 250 == 0:
        print(f"  sent {count} cmds...")
    time.sleep(0.02)

print(f"Done. Sent {count} commands.")
print("Was the robot stiff/resisting?")
