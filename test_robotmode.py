"""Check available RobotMode values."""
from booster_robotics_sdk import RobotMode
print("RobotMode members:")
for attr in dir(RobotMode):
    if not attr.startswith('_'):
        val = getattr(RobotMode, attr)
        print(f"  {attr} = {val}")
