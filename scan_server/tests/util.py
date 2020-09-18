import os
import yaml
import numpy as np
from scan_server import Scan, CalDriftPlan


try:
    d0 = os.path.dirname(os.path.realpath(__file__))
except:
    d0 = os.getcwd()
ssrl_dir = os.path.join(d0, "data", "ssrl_10_1_scan")

with open(os.path.join(ssrl_dir, "20200219_016_log"), "r") as f:
    d16 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open(os.path.join(ssrl_dir, "20200219_017_log"), "r") as f:
    d17 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open(os.path.join(ssrl_dir, "20200219_018_log"), "r") as f:
    d18 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open(os.path.join(ssrl_dir, "20200219_030_log"), "r") as f:
    d30 = list(yaml.load_all(f, Loader=yaml.SafeLoader))


def scan_logs_raw():
    return d16, d17, d18, d30

def scan_from_log(log: dict):
    "make Scan objects from .yaml logs from Jamie's software"
    scan = Scan(var_name="mono", var_unit="eV", scan_num=log["header"]["pass"], beamtime_id=0, 
                ext_id=log["header"]["htxs"], sample_id=0, sample_desc=d17[0]["sample"], extra={},
                data_path="no actual data", cal_drift_plan=CalDriftPlan(-1, "test", "test"))
    for i, mono_val in enumerate(log["mono"].keys()):
        start, end = log["mono"][mono_val]
        scan.point_start(mono_val, start, extra={})
        scan.point_end(end)
    scan.end()
    return scan

scan_logs = d17[1:] + d18[1:]
scan0_mono_vals = list(d17[1]["mono"].keys())
for scan_log in scan_logs:
    assert all(np.array(list(scan_log["mono"].keys())) == np.array(scan0_mono_vals))
_scans = [scan_from_log(log) for log in scan_logs]

def scans():
    return _scans