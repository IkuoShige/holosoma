from booster_robotics_sdk import ChannelFactory, B1LowStateSubscriber
import netifaces as ni
import time

ip = ni.ifaddresses('usb_eth0')[ni.AF_INET][0]['addr']
print(f'Using IP: {ip}')
ChannelFactory.Instance().Init(0, ip)

def handler(msg):
    print('Got state message!', msg.imu_state.rpy)

sub = B1LowStateSubscriber(handler)
sub.InitChannel()
print('Waiting for state messages...')
time.sleep(5)
print('Done')
