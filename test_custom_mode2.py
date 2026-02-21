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

# Init
ChannelFactory.Instance().Init(0, ip)

# State
latest_state = [None]

def state_handler(msg):
    latest_state[0] = msg

sub = B1LowStateSubscriber(state_handler)
sub.InitChannel()

# Wait for state with retry
print("Waiting for state...")
for i in range(50):  # up to 5 seconds
    if latest_state[0] is not None:
        break
    time.sleep(0.1)

if latest_state[0]:
    msg = latest_state[0]
    positions = [msg.motor_state_serial[i].q for i in range(NUM_MOTORS)]
    print(f"Got state! Joint positions (first 6): {[f'{q:.3f}' for q in positions[:6]]}")
    print(f"IMU RPY: {msg.imu_state.rpy}")
else:
    print("WARNING: No state received after 5s")
    positions = None

# Publisher + Client
pub = B1LowCmdPublisher()
pub.InitChannel()
client = B1LocoClient()
client.Init()

print(f"ChangeMode(kCustom)...")
result = client.ChangeMode(RobotMode.kCustom)
print(f"ChangeMode result: {result}")
print(f"RobotMode.kCustom value: {RobotMode.kCustom}")
time.sleep(0.5)

# Send commands - use state positions if available, else defaults
if positions is None:
    # K1 default standing angles
    positions = [
        0.0, 0.0,  # head
        0.2, -1.35, 0.0, -0.5,  # left arm
        0.2, 1.35, 0.0, 0.5,  # right arm
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
    ]
    print("Using default positions (no state available)")

print(f"\nSending position-hold commands for 10 seconds (kp=50, kd=2)...")
print(">>> Try pushing the robot gently - it should resist if custom mode is active <<<")
end = time.time() + 10
count = 0
while time.time() < end:
    # Re-read state if available
    if latest_state[0]:
        msg = latest_state[0]
        positions = [msg.motor_state_serial[i].q for i in range(NUM_MOTORS)]

    cmd = LowCmd()
    cmd.cmd_type = LowCmdType.SERIAL
    motor_cmds = [MotorCmd() for _ in range(NUM_MOTORS)]
    for i in range(NUM_MOTORS):
        motor_cmds[i].q = positions[i]
        motor_cmds[i].dq = 0.0
        motor_cmds[i].tau = 0.0
        motor_cmds[i].kp = 50.0
        motor_cmds[i].kd = 2.0
    cmd.motor_cmd = motor_cmds
    pub.Write(cmd)
    count += 1
    if count % 250 == 1:
        print(f"  sent {count} cmds, motor[0].q={positions[0]:.3f}, motor[10].q={positions[10]:.3f}")
    time.sleep(0.02)

print(f"Done. Sent {count} commands total.")
print("Was the robot stiff/resisting?")
