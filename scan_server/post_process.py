from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
import os
import glob
from collections import OrderedDict
import scan_server
from pathlib import Path
import tempfile
import shutil
import pylab as plt
import numpy as np
import argparse
import time


def get_scans_and_calibrations(beamtime_dir):
    # this depends on the files generated by test_tesscanner.py existing
    scan_filenames = glob.glob(os.path.join(beamtime_dir, "logs", "scan*.json"))
    cal_filenames = glob.glob(os.path.join(beamtime_dir, "logs", "calibration*.json"))
    scans = [scan_server.Scan.from_file(f) for f in sorted(scan_filenames)]
    calibrations = [scan_server.CalibrationLog.from_file(f) for f in sorted(cal_filenames)]
    scans = OrderedDict((scan.scan_num ,scan) for scan in scans)
    calibrations = OrderedDict((cal.cal_number, cal) for cal in calibrations)
    return scans, calibrations

def should_process(scan, calibrations):
    return needed_calibrations_exist(scan, calibrations) and (not output_exists(scan))

def output_exists(scan):
    return os.path.isdir(scan.user_output_dir)

def needed_calibrations_exist(scan, calibrations):
    if scan.drift_correction_plan == "none":
        return True
    if scan.drift_correction_plan == "basic":
        return True
    if scan.drift_correction_plan == "before_and_after_cals":
        return scan.previous_cal_log.cal_number+1 in calibrations.keys()
    else:
        raise Exception(f"scan drift_correct_strategy={scan.drift_correction_plan} is not recognized")

def get_scans_to_process(scans, calibrations):
    scans2 = OrderedDict((scan.scan_num, scan) for scan in scans.values() if should_process(scan, calibrations))
    return scans2

def calibration_apply_routine(routine, cal_number, data):
    # print(f"{routine=} {cal_number=}")
    routine = routines.get(routine)
    data.refreshFromFiles()
    return routine(cal_number, data)

def process(scan, calibrations, drift_correction_plan, bin_edges_ev, output_dir, max_channels=10000):
    assert not os.path.isdir(output_dir)
    data = ChannelGroup(getOffFileListFromOneFile(scan.data_path, maxChans=max_channels))
    cal = scan.previous_cal_log
    if drift_correction_plan == "basic":
        drift_correct_states = [f"CAL{cal.cal_number}", f"SCAN{scan.scan_num}"]
        data.learnDriftCorrection(states=drift_correct_states)
        attr = "filtValueDC"
    elif drift_correction_plan == "before_and_after_cals":
        drift_correct_states = [f"CAL{cal.cal_number}", f"SCAN{scan.scan_num}", f"CAL{cal.cal_number + 1}"]
        data.learnDriftCorrection(states=drift_correct_states)      
        attr = "filtValueDC"
    elif drift_correction_plan == "none":
        attr = "filtValue"
    else:
        raise Exception(f"drift_correction_plan={drift_correction_plan} not supported")

    routine = scan_server.routines.get(cal.routine)
    routine(cal.cal_number, data, attr=attr, calibratedName="energy")
    scan_hist2d = scan.hist2d(data, bin_edges_ev, "energy")
    temp_dir = tempfile.mktemp() # write outputs to a temporary dir, then move them to the location
    Path(temp_dir).mkdir(parents=False, exist_ok=False) 
    scan_hist2d.to_hdf5_file(os.path.join(temp_dir, f"scan{scan.scan_num:04d}_hist2d.hdf5"))
    fig = scan_hist2d.plot()
    fname = os.path.join(temp_dir, f"scan{scan.scan_num:04d}_hist2d.png")
    fig.savefig(fname)
    plt.close(fig)
    scan.to_disk(os.path.join(temp_dir, f"scan{scan.scan_num:04d}.json"))
    shutil.move(temp_dir, output_dir) # move all output to output_dir
    return scan_hist2d

def _find_scans_then_process_if_needed_and_ready(beamtime_dir, max_channels=10000):
    scans, calibrations = get_scans_and_calibrations(beamtime_dir)
    scans_to_process = get_scans_to_process(scans, calibrations)

    for scan in scans_to_process.values():
        print(f"processing: {scan}")
        process(scan, calibrations, scan.drift_correction_plan, 
            np.arange(0, 1000, 1), scan.user_output_dir,
            max_channels)

def post_process_script():
    parser = argparse.ArgumentParser(
        description="process TES data written by scan_server")
    parser.add_argument(
        "beamtime_dir", help="path to the user output beamtime directory", type=str)
    parser.add_argument(
        "--max_channels", help="max_channels to use when processing, mostly to speed up tests",
        type=int, default=10000)
    args = parser.parse_args()
    print(f"post_process_script started at {scan_server.rpc_server.time_human()}")
    tstart = time.time()
    _find_scans_then_process_if_needed_and_ready(args.beamtime_dir, args.max_channels)
    elapsed_s = time.time()-tstart
    print(f"post_process_script finished in {elapsed_s:.2f} s")