"""Reproduce EXACT full app init sequence to find what breaks ChangeMode."""
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


def check_light(label):
    input(f"\n>>> {label} - Is the light BLUE? (press Enter to continue) ")


# === Step 1: ChannelFactory (same as BasePolicy._init_sdk_components) ===
ChannelFactory.Instance().Init(0, ip)
print("[1] ChannelFactory.Init(0, ip) done")

# === Step 2: Subscriber (same as state_processor) ===
got_state = [False]
def state_handler(msg):
    got_state[0] = True
sub = B1LowStateSubscriber(state_handler)
sub.InitChannel()
print("[2] B1LowStateSubscriber created + InitChannel")

# === Step 3: Command sender init (same as booster_command_sender) ===
pub = B1LowCmdPublisher()
client = B1LocoClient()
pub.InitChannel()
client.Init()
print("[3] B1LowCmdPublisher + B1LocoClient created + Init")

# === Step 4: Prepare cmd + send (same as booster_command_sender) ===
low_cmd = LowCmd()
low_cmd.cmd_type = LowCmdType.SERIAL
motor_cmds = [MotorCmd() for _ in range(NUM_MOTORS)]
for i in range(NUM_MOTORS):
    motor_cmds[i].q = 0.0
    motor_cmds[i].dq = 0.0
    motor_cmds[i].tau = 0.0
    motor_cmds[i].kp = 0.0  # kp=0 because ONNX not loaded yet
    motor_cmds[i].kd = 0.0
    motor_cmds[i].weight = 0.0
low_cmd.motor_cmd = motor_cmds
pub.Write(low_cmd)
print("[4] Sent prepare cmd (kp=0, kd=0)")

# === Step 5: ChangeMode ===
result = client.ChangeMode(RobotMode.kCustom)
print(f"[5] ChangeMode(kCustom) result: {result}")
check_light("After ChangeMode")

# === Step 6: Simulate init delay (ONNX loading etc) ===
print("[6] Simulating 2s init delay (ONNX loading)...")
time.sleep(2)
check_light("After 2s delay")

# === Step 7: Send real commands ===
print("[7] Sending commands with kp=50, kd=2...")
for i in range(500):  # 10 seconds
    cmd = LowCmd()
    cmd.cmd_type = LowCmdType.SERIAL
    mc = [MotorCmd() for _ in range(NUM_MOTORS)]
    for j in range(NUM_MOTORS):
        mc[j].q = 0.0
        mc[j].dq = 0.0
        mc[j].tau = 0.0
        mc[j].kp = 50.0
        mc[j].kd = 2.0
    cmd.motor_cmd = mc
    pub.Write(cmd)
    time.sleep(0.02)
    if i % 250 == 0:
        print(f"  sent {i} cmds...")

check_light("After sending 500 commands")
print("Done.")
