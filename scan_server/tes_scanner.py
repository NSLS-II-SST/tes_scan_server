from statemachine import StateMachine, State
from typing import List, Any
import numpy as np
import time
from . import routines
import mass
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import io
import pylab as plt
import os
from pathlib import Path
from . import mass_monkey_patch



@dataclass_json
@dataclass
class CalibrationLog():
    start_unixnano: int
    end_unixnano: int
    off_filename: str
    sample_id: int
    sample_desc: str
    beamtime_id: str
    routine: str
    cal_number: int

    def to_disk(self, filename):
        assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json(indent=2))

@dataclass_json
@dataclass
class Scan():
    var_name: str
    var_unit: str
    scan_num: int
    beamtime_id: str
    sample_id: int
    sample_desc: str
    extra: dict
    data_path: str
    previous_cal_log: CalibrationLog
    drift_correction_plan: str
    point_extras: List[dict] = field(default_factory=list)
    var_values: List[float] = field(default_factory=list)
    epoch_time_start_s: List[int] = field(default_factory=list)
    epoch_time_end_s: List[int] = field(default_factory=list)
    _ended: bool = False

    def point_start(self, scan_var, epoch_time_s, extra):
        assert not self._ended
        assert len(self.epoch_time_start_s) == len(self.epoch_time_end_s)
        assert isinstance(extra, dict)
        self.var_values.append(float(scan_var))
        self.point_extras.append(extra)
        self.epoch_time_start_s.append(epoch_time_s)

    def point_end(self, epoch_time_s):
        assert len(self.epoch_time_start_s) - 1 == len(self.epoch_time_end_s)
        self.epoch_time_end_s.append(epoch_time_s)

    def write_experiment_state_file(self, f, header):
        if header:
            f.write("# unixnano, state label\n")
        for i, (start, end) in enumerate(zip(self.epoch_time_start_s, self.epoch_time_end_s)):
            label =  f"SCAN{self.scan_num}_{i}"
            f.write(f"{int(start*1e9)}, {label}\n")
            f.write(f"{int(end*1e9)}, PAUSE\n")

    def experiment_state_file_as_str(self, header):
        with io.StringIO() as f:
            self.write_experiment_state_file(f, header)
            return f.getvalue()

    def end(self):
        assert not self._ended
        self._ended = True

    def description_str(self):
        return f"scan{self.scan_num} sample{self.sample_id} beamtime_id{self.beamtime_id}"

    def __repr__(self):
        return f"<Scan num{self.scan_num} beamtime_id{self.beamtime_id} ext_id{self.ext_id} npts{len(self.var_values)}"

    def hist2d(self, data, bin_edges, attr):
        starts_nano = (1e9*np.array(self.epoch_time_start_s)).astype(int)
        ends_nano = (1e9*np.array(self.epoch_time_end_s)).astype(int)
        assert len(starts_nano) == len(ends_nano)
        var_name = self.var_name
        var_unit = self.var_unit
        var_vals = self.var_values
        hist2d = np.zeros((len(bin_edges)-1, len(starts_nano)))
        for ds in data.values():
            states = ds.unixnanoToStates(starts_nano, ends_nano)
            for i, _state in enumerate(states):
                bin_centers, counts = ds.hist(bin_edges, attr, states=_state)
                hist2d[:, i] += counts
        return ScanHist2DResult(hist2d, var_name, var_unit, var_vals, bin_centers, attr, "eV", self.description_str(), data.shortName)

    def to_disk(self, filename):
        assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json(indent=2))


@dataclass
class ScanHist2DResult():
    hist2d: Any # really np.ndarray
    var_name: str
    var_unit: str
    var_vals: Any # really np.nda
    bin_centers: Any # really np.ndarray
    attr: str
    attr_unit: str
    scan_desc: str
    pulse_file_desc: str

    def plot(self):
        plt.figure()
        plt.contourf(self.var_vals, self.bin_centers, self.hist2d, cmap="gist_heat")
        plt.xlabel(f"{self.var_name} ({self.var_unit})")
        plt.ylabel(f"{self.attr} ({self.attr_unit})")
        plt.title(self.scan_desc)

    def __add__(self, other):
        assert self.var_name == other.var_name
        assert self.var_unit == other.var_unit
        assert all([a == b for (a, b) in zip(self.var_vals, other.var_vals)])
        assert all([a == b for (a, b) in zip(self.bin_centers, other.bin_centers)])
        assert self.attr == other.attr
        assert self.attr_unit == other.attr_unit
        assert self.pulse_file_desc == other.pulse_file_desc
        return ScanHist2DResult(self.hist2d+other.hist2d, self.var_name, self.var_unit, self.var_vals,
        self.bin_centers, self.attr, self.attr_unit, "sum scan", self.pulse_file_desc)

class ScannerState(StateMachine):
    """defines allowed state transitions, transitions will error if you do an invalid one"""
    no_file = State('no_file', initial=True)
    pause = State("pause")
    scan = State('scan')
    scan_point = State("scan_point")
    cal_data = State('cal_data')

    scan_start = pause.to(scan)
    scan_end = scan.to(pause)

    scan_point_start = scan.to(scan_point)
    scan_point_end = scan_point.to(scan)

    file_start = no_file.to(pause)
    file_end = pause.to(no_file)

    cal_data_start = pause.to(cal_data)
    cal_data_end = cal_data.to(pause)


class TESScanner():
    """talks to dastard and mass"""
    def __init__(self, dastard, beamtime_id: str, base_user_output_dir: str):
        self.dastard = dastard
        self.beamtime_id = beamtime_id
        self.base_user_output_dir = base_user_output_dir
        self.state: ScannerState = ScannerState()
        self.reset()

    def reset(self):
        self.last_scan = None
        self.scan = None
        self.data = None
        self.roi_counts_start_unixnano = None
        self.rois_bin_edges = None
        self.calibration_to_routine: List[str] = [] 
        self.next_cal_number: int = 0
        self.calibration_log = None
        self.off_filename = None

    def file_start(self):
        self.state.file_start()
        self.off_filename = self.dastard.start_file()
        self.data = ChannelGroup(getOffFileListFromOneFile(self.off_filename))

    def calibration_data_start(self, sample_id: int, sample_desc: str, routine: str):
        self.state.cal_data_start()
        self.dastard.set_pulse_triggers()
        self.dastard.set_experiment_state(f"CAL{self.next_cal_number}")
        self.calibration_to_routine.append(routine)
        self.calibration_log = CalibrationLog(start_unixnano=time_unixnano(),
            end_unixnano=None, off_filename=self.off_filename,
            sample_id = sample_id, sample_desc = sample_desc, beamtime_id = self.beamtime_id, routine = routine, 
            cal_number = self.next_cal_number)
        self.next_cal_number += 1
        assert len(self.calibration_to_routine) == self.next_cal_number

    def calibration_data_end(self):
        self.state.cal_data_end()
        self.dastard.set_experiment_state("PAUSE")
        self.calibration_log.end_unixnano = time_unixnano()
        for fname in self.log_filenames("calibration", self.calibration_log.cal_number):
            self.calibration_log.to_disk(fname)

    def calibration_learn_from_last_data(self):
        self._calibration_apply_routine(self.calibration_log.routine, 
            self.calibration_log.cal_number, self.data)
        # now we can access energy

    def _calibration_apply_routine(self, routine, cal_number, data):
        # print(f"{routine=} {cal_number=}")
        routine = routines.get(routine)
        data.refreshFromFiles()
        return routine(cal_number, data)

    def roi_set(self, rois_list):
        # roi list is a a list of pairs of lo, hi energy pairs
        assert len(rois_list) > 0
        bin_edges = []
        for (lo_ev, hi_ev) in rois_list:
            assert hi_ev > lo_ev
            if len(bin_edges) > 0:
                assert lo_ev > bin_edges[-1]
            bin_edges.append(lo_ev)
            bin_edges.append(hi_ev)
        self.rois_bin_edges = np.array(bin_edges)
    
    def roi_start_counts(self):
        self.roi_counts_start_unixnano = time_unixnano()
    
    def roi_get_counts(self):
        "first call set_rois, then start_rois_counts, then this will return counts in all rois since last call to start_rois_counts"
        assert self.roi_counts_start_unixnano is not None, "first call set_rois, then start_rois_counts, roi_start_counts, then roi_get_counts"
        assert self.rois_bin_edges is not None, "rois_bin_edges is None: first call set_rois, then start_rois_counts, roi_start_counts, then roi_get_counts"
        a, b = self.roi_counts_start_unixnano, time_unixnano()
        self.roi_counts_start_unixnano = None
        bin_centers, counts = self.data.histWithUnixnanos(self.rois_bin_edges, "energy", [a], [b])
        return counts[::2]

    def scan_start(self, var_name, var_unit, scan_num, sample_id, sample_desc, extra, drift_correction_plan):
        assert isinstance(scan_num, int)
        for fname in self.log_filenames("scan", scan_num):
            assert not os.path.isfile(fname)
        self.state.scan_start()
        data_path = self.dastard.get_data_path()
        self.validate_drift_correction_plan(drift_correction_plan)
        self.scan = Scan(var_name, var_unit, scan_num, self.beamtime_id, sample_id, 
            sample_desc, extra, data_path,
            previous_cal_log=self.calibration_log,
            drift_correction_plan=drift_correction_plan)
        self.dastard.set_experiment_state(f"SCAN{scan_num}")

    def scan_point_start(self, scan_var, extra):
        self.state.scan_point_start()
        epoch_time_s = time.time()
        self.scan.point_start(scan_var, epoch_time_s, extra)

    def scan_point_end(self):
        self.state.scan_point_end()
        epoch_time_s = time.time()
        self.scan.point_end(epoch_time_s)

    def scan_end(self):
        self.state.scan_end()
        self.scan.end()
        for fname in self.log_filenames("scan", self.scan.scan_num):
            self.scan.to_disk(fname)
        self.last_scan = self.scan
        self.scan = None
        self.dastard.set_experiment_state("PAUSE")
        
    def scan_start_calc_last_outputs(self):
        scan_hist2d = self.last_scan.hist2d(self.data, np.arange(0, 1000, 1), "energy")
        scan_hist2d.plot()
        scan_num = self.last_scan.scan_num
        fname = os.path.join(self.scan_user_output_dir(scan_num, "plots"), f"rt_{scan_num}.png")
        plt.savefig(fname)
        plt.close()
        print(fname)
        # TODO.... make this async and make it do higher quality analysis
        # self.launch_process_to_calc_outputs(drift_correction_plan)
        return

    def file_end(self):
        self.state.file_end()
        self.reset()
    
    def beamtime_user_output_dir(self, subdir = None):
        dirname = os.path.join(self.base_user_output_dir, f"beamtime_{self.beamtime_id}")
        if subdir is not None:
            dirname = os.path.join(dirname, subdir)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

    def scan_user_output_dir(self, scan_num, subdir = None):
        dirname = self.beamtime_user_output_dir(f"scan{scan_num:4d}")
        if subdir is not None:
            dirname = os.path.join(dirname, subdir)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname
        
    def user_log_dir(self):
        return self.beamtime_user_output_dir("logs")

    def tes_log_dir(self):
        dirname = os.path.join(os.path.dirname(self.off_filename),"logs")
        Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

    def log_filenames(self, log_name, log_num):
        # we duplicate logs
        # one set goes to the beamtime directory for user consumption
        # another set lives with the off files for convenience
        assert log_name in ["scan", "calibration"]
        filename1 = os.path.join(self.user_log_dir(), f"{log_name}{log_num:04d}.json")
        filename2 = os.path.join(self.tes_log_dir(), f"{log_name}{log_num:04d}.json")
        assert not os.path.isfile(filename1), f"{filename1} already exists"
        assert not os.path.isfile(filename2), f"{filename2} already exists"
        return [filename1, filename2]        

    def validate_drift_correction_plan(self, drift_correction_plan):
        if drift_correction_plan not in ["testing_not_real"]:
            raise Exception("invalid drift plan")

def time_unixnano():
    return int(1e9*time.time())

