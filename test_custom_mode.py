"""Test ChangeMode(kCustom) and verify robot responds to commands."""
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

# Step 1: Init ChannelFactory
ChannelFactory.Instance().Init(0, ip)
print("[1] ChannelFactory initialized")

# Step 2: Create subscriber
joint_positions = []


def state_handler(msg):
    joint_positions.clear()
    for i in range(NUM_MOTORS):
        joint_positions.append(msg.motor_state_serial[i].q)


sub = B1LowStateSubscriber(state_handler)
sub.InitChannel()
print("[2] Subscriber created")
time.sleep(1)
print(f"    State receiving: {'YES' if joint_positions else 'NO'}")
if joint_positions:
    print(f"    Joint positions: {[f'{q:.3f}' for q in joint_positions[:6]]}...")

# Step 3: Create publisher
pub = B1LowCmdPublisher()
pub.InitChannel()
print("[3] Publisher created")

# Step 4: Create client and Init
client = B1LocoClient()
client.Init()
print("[4] B1LocoClient initialized")

# Step 5: ChangeMode
print("[5] Calling ChangeMode(kCustom)...")
result = client.ChangeMode(RobotMode.kCustom)
print(f"    ChangeMode result: {result}")
time.sleep(0.5)

# Step 6: Send commands - hold current position with stiffness
print("[6] Sending position-hold command with kp=50, kd=2...")
if joint_positions:
    for trial in range(3):
        cmd = LowCmd()
        cmd.cmd_type = LowCmdType.SERIAL
        motor_cmds = [MotorCmd() for _ in range(NUM_MOTORS)]
        for i in range(NUM_MOTORS):
            motor_cmds[i].q = joint_positions[i] if i < len(joint_positions) else 0.0
            motor_cmds[i].dq = 0.0
            motor_cmds[i].tau = 0.0
            motor_cmds[i].kp = 50.0  # Moderate stiffness
            motor_cmds[i].kd = 2.0
        cmd.motor_cmd = motor_cmds
        pub.Write(cmd)
        time.sleep(0.02)

    print("    Commands sent. Robot should feel stiff if in custom mode.")
    print("    Try gently pushing the robot - does it resist?")

    # Keep sending for 5 seconds
    print("    Sending for 5 seconds...")
    end = time.time() + 5
    count = 0
    while time.time() < end:
        cmd = LowCmd()
        cmd.cmd_type = LowCmdType.SERIAL
        motor_cmds_loop = [MotorCmd() for _ in range(NUM_MOTORS)]
        for i in range(NUM_MOTORS):
            motor_cmds_loop[i].q = joint_positions[i] if i < len(joint_positions) else 0.0
            motor_cmds_loop[i].dq = 0.0
            motor_cmds_loop[i].tau = 0.0
            motor_cmds_loop[i].kp = 50.0
            motor_cmds_loop[i].kd = 2.0
        cmd.motor_cmd = motor_cmds_loop
        pub.Write(cmd)
        count += 1
        time.sleep(0.02)  # 50Hz
    print(f"    Sent {count} commands in 5s")
else:
    print("    No state data - cannot send commands")

print("\nDone. Was the robot stiff/resisting movement?")
