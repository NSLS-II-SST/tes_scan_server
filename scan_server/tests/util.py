import os
import yaml
import numpy as np
from scan_server import DataScan
import shutil


try:
    d0 = os.path.dirname(os.path.realpath(__file__))
except:
    d0 = os.getcwd()
ssrl_dir = os.path.join(d0, "data", "ssrl_10_1_scan")
ssrl_filename_pattern = os.path.join(ssrl_dir, "20200219_%s.%s")

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
    scan = DataScan(var_name="mono", var_unit="eV", scan_num=log["header"]["htxs"], beamtime_id="test",
                sample_id=0, sample_desc=d17[0]["sample"], extra={}, user_output_dir="dummy",
                data_path="no actual data", drift_correction_plan = None)
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

def make_states_to_unixnano():
    _states_to_unixnano = {"PAUSE" : []}
    cal_start, cal_stop = int(d16[1]["header"]["start"]*1e9), int(d16[1]["header"]["stop"]*1e9)
    _states_to_unixnano[f"CAL0"] = cal_start
    _states_to_unixnano["PAUSE"].append(cal_stop)
    for j, scan in enumerate(scans()):
        _states_to_unixnano[f"SCAN{j}"] = int(scan.epoch_time_start_s[0]*1e9)
        _states_to_unixnano["PAUSE"].append(int(scan.epoch_time_end_s[-1]*1e9))
    cal_start, cal_stop = int(d30[1]["header"]["start"]*1e9), int(d30[1]["header"]["stop"]*1e9)
    _states_to_unixnano[f"CAL1"] = cal_start
    _states_to_unixnano["PAUSE"].append(cal_stop)
    return _states_to_unixnano

_states_to_unixnano = make_states_to_unixnano()

def write_ssrl_experiment_state_file(filename):
    with open(filename, "w") as f:
        f.write("# unixnano, label\n")
        f.write("0, START\n")
        cal_start, cal_stop = int(d16[1]["header"]["start"]*1e9), int(d16[1]["header"]["stop"]*1e9)
        f.write(f"{cal_start}, CAL0\n")
        f.write(f"{cal_stop}, PAUSE\n")
        for j, scan in enumerate(scans()):
            f.write(f"{int(scan.epoch_time_start_s[0]*1e9)}, SCAN{j}\n")
            f.write(f"{int(scan.epoch_time_end_s[-1]*1e9)}, PAUSE\n")
        cal_start, cal_stop = int(d30[1]["header"]["start"]*1e9), int(d30[1]["header"]["stop"]*1e9)
        f.write(f"{cal_start}, CAL1\n")
        f.write(f"{cal_stop}, PAUSE\n")


import os, errno

def silentremovedir(dirname):
    try:
        shutil.rmtree(dirname)
    except FileNotFoundError:
        pass

def pre_test_cleanup():
    # delete files previous tests create
    silentremovedir(os.path.join(ssrl_dir, "logs"))
    silentremovedir(os.path.join(ssrl_dir, "base_user_output_dir"))

pre_test_cleanup()

