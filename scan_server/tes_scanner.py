from statemachine import StateMachine, State
from typing import List, Any
import numpy as np
import time
from . import routines
import mass
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
from dataclasses import dataclass, field, fields
from dataclasses_json import dataclass_json
import io
import pylab as plt
import os
from pathlib import Path
from . import mass_monkey_patch
import h5py
from scan_server import post_process
import subprocess



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

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rb") as f:
            return cls.from_json(f.read())

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
    user_output_dir: str
    # could add roi_list
    previous_cal_log: CalibrationLog
    drift_correction_plan: str
    point_extras: dict = field(default_factory=dict)
    var_values: List[float] = field(default_factory=list)
    epoch_time_start_s: List[int] = field(default_factory=list)
    epoch_time_end_s: List[int] = field(default_factory=list)
    _ended: bool = False

    def point_start(self, scan_var, epoch_time_s, extra=None):
        assert not self._ended
        assert len(self.epoch_time_start_s) == len(self.epoch_time_end_s)
        assert isinstance(extra, dict)
        self.var_values.append(float(scan_var))
        if extra is not None and extra != {}:
            idx = str(len(self.epoch_time_start_s))
            self.point_extras[idx] = extra
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
        return f"<Scan num{self.scan_num} beamtime_id{self.beamtime_id} npts{len(self.var_values)}>"

    def hist2d(self, data, bin_edges, attr):
        starts_nano = int(1e9)*np.array(self.epoch_time_start_s, dtype="int64")
        ends_nano = int(1e9)*np.array(self.epoch_time_end_s, dtype="int64")
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

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rb") as f:
            return cls.from_json(f.read())


@dataclass
class ScanHist2DResult():
    hist2d: Any # really np.ndarray
    var_name: str
    var_unit: str
    var_vals: Any # really np.ndarray
    bin_centers: Any # really np.ndarray
    attr: str
    attr_unit: str
    scan_desc: str
    pulse_file_desc: str

    def plot(self):
        fig = plt.figure()
        plt.contourf(self.var_vals, self.bin_centers, self.hist2d, cmap="gist_heat")
        plt.xlabel(f"{self.var_name} ({self.var_unit})")
        plt.ylabel(f"{self.attr} ({self.attr_unit})")
        plt.title(self.scan_desc)
        return fig

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

    def to_hdf5_file(self, filename):
        assert not os.path.isfile(filename)
        with h5py.File(filename, "w") as h5:
            self.to_hdf5(h5)     

    def to_hdf5(self, h5):
        for field in fields(self):
            h5[field.name] = getattr(self, field.name)
    
    @classmethod
    def from_hdf5(cls, h5):
        args = [h5[field.name] for field in fields(cls)]
        return cls(*args)

    @classmethod
    def from_hdf5_file(cls, filename):
        with h5py.File(filename, "r") as h5:
            return cls.from_hdf5(h5)

class ScannerState(StateMachine):
    """defines allowed state transitions, transitions will error if you do an invalid one"""
    no_file = State('no_file', initial=True)
    file_open = State("file_open")
    scan = State('scan')
    scan_point = State("scan_point")
    cal_data = State('cal_data')

    scan_start = file_open.to(scan)
    scan_end = scan.to(file_open)

    scan_point_start = scan.to(scan_point)
    scan_point_end = scan_point.to(scan)

    file_start = no_file.to(file_open)
    file_end = file_open.to(no_file)

    cal_data_start = file_open.to(cal_data)
    cal_data_end = cal_data.to(file_open)


class TESScanner():
    """talks to dastard and mass"""
    def __init__(self, dastard, beamtime_id: str, base_user_output_dir: str, background_process_log_file):
        self.dastard = dastard
        self.beamtime_id = beamtime_id
        self.base_user_output_dir = base_user_output_dir
        self.state: ScannerState = ScannerState()
        self._reset()
        self.background_process = None # dont put this in _reset, we want it to persist over reset
        self.background_process_log_file = background_process_log_file


    def _reset(self):
        self.last_scan = None
        self.scan = None
        self._data = None
        self.roi_counts_start_unixnano = None
        self.rois_bin_edges = None
        self.calibration_to_routine: List[str] = [] 
        self.next_cal_number: int = 0
        self.calibration_log = None
        self.off_filename = None

    def file_start(self, ljh22, off, path=None):
        """tell dastard to start a new file, must be called before any calibration or scan functions"""
        self.state.file_start()
        self.off_filename = self.dastard.start_file(ljh22, off, path)
        # dastard lazily creates off files when it has data to write
        # so we need to wait to open the off files until some time has
        # passed from calling file_start
        # so we must always access through _get_data (_hides it from the rpc)
    
    def _get_data(self):
        assert self.off_filename is not None, "self.off_filename is None"
        if self._data is None:
            self._data = ChannelGroup(getOffFileListFromOneFile(self.off_filename))
        return self._data

    def calibration_data_start(self, sample_id: int, sample_desc: str, routine: str):
        """start taking calibraion data, ensure the appropriate x-rays are incident on the detector
        sample_id: int - for your reference
        sample_desc: str - for your reference
        routine: str - which function is used to generate calibration curves from the data
        """
        self.state.cal_data_start()
        # self.set_pulse_triggers()
        self.dastard.set_experiment_state(f"CAL{self.next_cal_number}")
        self.calibration_to_routine.append(routine)
        self.calibration_log = CalibrationLog(start_unixnano=time_unixnano(),
            end_unixnano=None, off_filename=self.off_filename,
            sample_id = sample_id, sample_desc = sample_desc, beamtime_id = self.beamtime_id, routine = routine, 
            cal_number = self.next_cal_number)
        self.next_cal_number += 1
        assert len(self.calibration_to_routine) == self.next_cal_number

    def calibration_data_end(self, _try_post_processing=True):
        """stop taking calabration data, will write a log file"""
        self.state.cal_data_end()
        self.dastard.set_experiment_state("PAUSE")
        self.calibration_log.end_unixnano = time_unixnano()
        for fname in self._log_filenames("calibration", self.calibration_log.cal_number):
            self.calibration_log.to_disk(fname)
        if _try_post_processing:
            self.start_post_processing()

    def calibration_learn_from_last_data(self):
        """use the last calibration data plus the specified routine to learn the realtime energy calibration curves"""
        #INTENT: Nothing can prevent data taking, and we should be able to manually
        # fix calibration somehow.
        # 1. we should have a "failed calibration" state that allow data taking to proceed,
        # all functionatily should work in this state, but we can return 0 for roi_counts
        # we should always have a "TFY" ROI, so that can return true values during failed calibration
        # 2. There should be a way to "fix" the calibration, either a pickle file that we can write to
        # or a exposed RPC method you can call to register a new calibration or something.
        self._calibration_apply_routine(self.calibration_log.routine, 
            self.calibration_log.cal_number, self._get_data())
        # now we can access energy

    def _calibration_apply_routine(self, routine, cal_number, data):
        # print(f"{routine=} {cal_number=}")
        routine = routines.get(routine)
        data.refreshFromFiles()
        return routine(cal_number, data, attr="filtValue", calibratedName="energy")

    # could take (name, hi, lo)
    def roi_set(self, rois_list: List):
        """must be alled before other roi functions
        rois_list: a list of (lo, hi) energy pairs in eV, each pair specifies a region of interest""" 
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
        """take a timestamp for future reference"""
        self.roi_counts_start_unixnano = time_unixnano()
    
    def roi_get_counts(self):
        """return a list of counts in each ROI since roi_start_counts was called
        must always call roi_start_counts and roi_get_counts in pairs"""
        assert self.roi_counts_start_unixnano is not None, "first call set_rois, then start_rois_counts, roi_start_counts, then roi_get_counts"
        assert self.rois_bin_edges is not None, "rois_bin_edges is None: first call set_rois, then start_rois_counts, roi_start_counts, then roi_get_counts"
        a, b = self.roi_counts_start_unixnano, time_unixnano()
        self.roi_counts_start_unixnano = None
        bin_centers, counts = self._get_data().histWithUnixnanos(self.rois_bin_edges, "energy", [a], [b])
        return counts[::2]

    def scan_start(self, var_name: str, var_unit: str, scan_num: int, sample_id: int, sample_desc: str, extra: dict, drift_correction_plan: str):
        assert isinstance(scan_num, int)
        for fname in self._log_filenames("scan", scan_num):
            assert not os.path.isfile(fname)
        self.state.scan_start()
        data_path = self.dastard.get_data_path()
        self._validate_drift_correction_plan(drift_correction_plan)
        user_output_dir = self._scan_user_output_dir(scan_num, make=False)
        self.scan = Scan(var_name, var_unit, scan_num, self.beamtime_id, sample_id, 
            sample_desc, extra, data_path,
            user_output_dir,
            previous_cal_log=self.calibration_log,
            drift_correction_plan=drift_correction_plan)
        # we could record the ROIs (name, lo, hi) so when we do post_process 
        # we can label the plots with the names 
        # quick post process would be called from TESScanner which knows about the rois
        self.dastard.set_experiment_state(f"SCAN{scan_num}")

    def scan_point_start(self, scan_var: float, extra: dict, _epoch_time_s_for_test=None):
        self.state.scan_point_start()
        if _epoch_time_s_for_test is None:
            epoch_time_s = time.time()
        else:
            epoch_time_s = _epoch_time_s_for_test
        self.scan.point_start(scan_var, epoch_time_s, extra)

    def scan_point_end(self, _epoch_time_s_for_test=None):
        self.state.scan_point_end()
        if _epoch_time_s_for_test is None:
            epoch_time_s = time.time()
        else:
            epoch_time_s = _epoch_time_s_for_test
        self.scan.point_end(epoch_time_s)

    def scan_end(self, _try_post_processing=True):
        self.state.scan_end()
        self.scan.end()
        for fname in self._log_filenames("scan", self.scan.scan_num):
            self.scan.to_disk(fname)
        self.last_scan = self.scan
        self.scan = None
        self.dastard.set_experiment_state("PAUSE")
        if _try_post_processing:
            self.start_post_processing()
        
    def start_post_processing(self, _wait_for_finish=False, _max_channels=10000):
        if self.background_process is not None:
            # we could remove this logic and let more than one background process run
            # it would be slightly faster
            # but first we would want to improve the logic on deciding which channels to process
            # right now it is based on the directory created when it is done, so if you start
            # two processes quickly, they will do the same work
            isdone = self.background_process.poll() is not None
            if not isdone:
                return "previous process still running"
        args = ["process_scans", self._beamtime_user_output_dir(), f"--max_channels={_max_channels}"]
        print(args)
        self.background_process = subprocess.Popen(args, stdout = self.background_process_log_file, stderr=subprocess.STDOUT)
        return "started new process"

    def file_end(self):
        self.state.file_end()
        self.dastard.stop_writing()
        self._reset()
    
    def _beamtime_user_output_dir(self, subdir = None, make = True):
        dirname = os.path.join(self.base_user_output_dir, f"beamtime_{self.beamtime_id}")
        if subdir is not None:
            dirname = os.path.join(dirname, subdir)
        if make:
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

    def _scan_user_output_dir(self, scan_num, subdir = None, make = True):
        dirname = self._beamtime_user_output_dir(f"scan{scan_num:04d}", make = make)
        if subdir is not None:
            dirname = os.path.join(dirname, subdir)
        if make:
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname
        
    def _user_log_dir(self):
        return self._beamtime_user_output_dir("logs")

    def _tes_log_dir(self):
        dirname = os.path.join(os.path.dirname(self.off_filename),"logs")
        Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

    def _log_filenames(self, log_name, log_num):
        # we duplicate logs
        # one set goes to the beamtime directory for user consumption
        # another set lives with the off files for convenience
        assert log_name in ["scan", "calibration"]
        filename1 = os.path.join(self._user_log_dir(), f"{log_name}{log_num:04d}.json")
        filename2 = os.path.join(self._tes_log_dir(), f"{log_name}{log_num:04d}.json")
        assert not os.path.isfile(filename1), f"{filename1} already exists"
        assert not os.path.isfile(filename2), f"{filename2} already exists"
        return [filename1, filename2]        

    def _validate_drift_correction_plan(self, drift_correction_plan):
        if drift_correction_plan not in ["none", "basic", "before_and_after_cals"]:
            raise Exception("invalid drift plan")

def time_unixnano():
    return int(1e9*time.time())

