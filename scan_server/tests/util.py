import os
import yaml

try:
    d0 = os.path.dirname(os.path.realpath(__file__))
except:
    d0 = os.getcwd()
d = os.path.join(d0, "data", "ssrl_10_1_scan")

with open(os.path.join(d,"20200219_016_log"),"r") as f:
    d16 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open(os.path.join(d,"20200219_017_log"),"r") as f:
    d17 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open(os.path.join(d,"20200219_018_log"),"r") as f:
    d18 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open(os.path.join(d,"20200219_030_log"),"r") as f:
    d30 = list(yaml.load_all(f, Loader=yaml.SafeLoader))


def scan_logs():
    return d16, d17, d18, d30