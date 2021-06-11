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
from os.path import join, exists, basename
from pathlib import Path
from . import mass_monkey_patch
import h5py
from scan_server import post_process
import subprocess
from glob import glob



@dataclass_json
@dataclass
class BaseScan():
    var_name: str
    var_unit: str
    scan_num: int
    beamtime_id: str
    sample_id: int
    sample_desc: str
    extra: dict
    data_path: str
    drift_correction_plan: str
    point_extras: dict = field(default_factory=dict)
    var_values: List[float] = field(default_factory=list)
    epoch_time_start_s: List[int] = field(default_factory=list)
    epoch_time_end_s: List[int] = field(default_factory=list)
    roi: dict = field(default_factory=dict)
    _ended: bool = field(default=False)

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

    def to_disk(self, filename, overwrite=False):
        if not overwrite:
            assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json(indent=2))

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rb") as f:
            return cls.from_json(f.read())

@dataclass_json
@dataclass
class DataScan(BaseScan):
    cal_number: int = -1
    calibration: bool = field(default=False)


@dataclass_json
@dataclass
class CalibrationScan(BaseScan):
    routine: str = "none"
    calibration: bool = field(default=True)


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
    #cal_data = State('cal_data')

    scan_start = file_open.to(scan)
    scan_end = scan.to(file_open)

    scan_point_start = scan.to(scan_point)
    scan_point_end = scan_point.to(scan)

    file_start = no_file.to(file_open)
    file_end = file_open.to(no_file)

    #cal_data_start = file_open.to(cal_data)
    #cal_data_end = cal_data.to(file_open)


class TESScanner():
    """talks to dastard and mass"""
    def __init__(self, dastard, beamtime_id: str, base_user_output_dir: str, background_process_log_file):
        self._dastard = dastard
        self.beamtime_id = beamtime_id
        self.base_user_output_dir = base_user_output_dir
        self._state: ScannerState = ScannerState()
        self.tfy_ulim = 1600
        self.tfy_llim = 200
        self._reset()
        self.background_process = None # dont put this in _reset, we want it to persist over reset
        self.background_process_log_file = background_process_log_file


    def _reset(self):
        self._last_scan = None
        self._scan = None
        self._data = None
        self.roi = None
        self._scan_num = None
        self.overwrite = False
        self.calibration_to_routine: List[str] = [] 
        self.calibration_state = "no_calibration"
        self.cal_number: int = -1
        self._off_filename = None


    # Properties
    @property
    def state(self):
        return self._state.current_state_value
    
    @property
    def filename(self):
        return self._off_filename
        

    def roi_get(self, key=None):
        if key is None:
            return self._roi
        else:
            return self._roi.get(key, None)

    def roi_set(self, roi_dict):
        """must be alled before other roi functions
        rois_list: a dictinary of {name: (lo, hi), ...} energy pairs in eV, each pair specifies a region of interest""" 
        # roi list is a a list of pairs of lo, hi energy pairs
        if roi_dict is None or len(roi_dict) == 0:
            self._roi = {"tfy": (self.tfy_llim, self.tfy_ulim)}
            return
        else:
            for key, (lo_ev, hi_ev) in roi_dict.items():
                if lo_ev is None or hi_ev is None:
                    self._roi.pop(key, None)
                    roi_dict.pop(key, None)
                assert hi_ev > lo_ev
            self._roi.update(roi_dict)
            return
        
    @property
    def scan_num(self):
        if self._scan_num is None:
            self._scan_num = self._get_current_scan_num_from_logs()    
        return self._scan_num        

    @property
    def next_scan_num(self):
        return self.scan_num + 1
    
    def _advance_scan_num(self):
        self._scan_num = self.scan_num + 1
        return self._scan_num
    
    # Dastard operations
    def file_start(self, path=None):
        """tell dastard to start a new file, must be called before any calibration or scan functions"""
        self._state.file_start()
        ljh = False
        off = True
        self._off_filename = self._dastard.start_file(ljh, off, path)
        return self._off_filename
        # dastard lazily creates off files when it has data to write
        # so we need to wait to open the off files until some time has
        # passed from calling file_start
        # so we must always access through _get_data (_hides it from the rpc)

    def file_end(self):
        self._state.file_end()
        self._dastard.stop_writing()
        self._reset()

    # Scan operations
    def scan_start(self, var_name: str, var_unit: str, scan_num: int, sample_id: int, sample_desc: str, extra: dict, drift_correction_plan: str):
        
        self._state.scan_start()
        scan_num = self.scan_num
        for fname in self._log_filenames("scan", scan_num):
            assert not os.path.isfile(fname)
        data_path = self._dastard.get_data_path()
        self._validate_drift_correction_plan(drift_correction_plan)
        self._scan = DataScan(var_name, var_unit, scan_num, self.beamtime_id, sample_id, 
            sample_desc, extra, data_path, drift_correction_plan=drift_correction_plan,
            cal_number=self.cal_number, roi=self.roi)
        self._dastard.set_experiment_state(f"SCAN{scan_num}")

        
    def scan_point_start(self, scan_var: float, extra: dict, _epoch_time_s_for_test=None):
        self._state.scan_point_start()
        if _epoch_time_s_for_test is None:
            epoch_time_s = time.time()
        else:
            epoch_time_s = _epoch_time_s_for_test
        self._scan.point_start(scan_var, epoch_time_s, extra)

    def scan_point_end(self, _epoch_time_s_for_test=None):
        self._state.scan_point_end()
        if _epoch_time_s_for_test is None:
            epoch_time_s = time.time()
        else:
            epoch_time_s = _epoch_time_s_for_test
        self._scan.point_end(epoch_time_s)

    def scan_end(self, _try_post_processing=True):
        self._state.scan_end()
        self._scan.end()
        scan_name = "calibration" if self._scan.calibration else "scan"
        for fname in self._log_filenames(scan_name, self._scan.scan_num):
            self._scan.to_disk(fname, self.overwrite)
        self._last_scan = self._scan
        self._advance_scan_num()
        self._scan = None
        self._dastard.set_experiment_state("PAUSE")
        if _try_post_processing:
            self.start_post_processing()

    # Calibration Functions
    def calibration_start(self, var_name: str, var_unit: str, scan_num: int, sample_id: int, sample_desc: str, extra: dict, drift_correction_plan: str, routine: str):
        """start taking calibraion data, ensure the appropriate x-rays are incident on the detector
        sample_id: int - for your reference
        sample_desc: str - for your reference
        routine: str - which function is used to generate calibration curves from the data
        """
        self._state.scan_start()
        # self.set_pulse_triggers()

        self.calibration_to_routine.append(routine)
        data_path = self._dastard.get_data_path()
        self._validate_drift_correction_plan(drift_correction_plan)
        scan_num = self.scan_num
        self._scan = CalibrationScan(var_name, var_unit, scan_num, self.beamtime_id, sample_id, 
                                    sample_desc, extra, data_path, 
                                    drift_correction_plan, routine=routine,
                                    roi=self.roi)
        self._dastard.set_experiment_state(f"CAL{scan_num}")
        self.cal_number = scan_num
        

    def calibration_learn_from_last_data(self):
        """use the last calibration data plus the specified routine to learn the realtime energy calibration curves"""
        self._calibration_apply_routine(self.calibration_to_routine[-1],
            self.cal_number, self._get_data())
        # now we can access energy

    def _calibration_apply_routine(self, routine, cal_number, data):
        # print(f"{routine=} {cal_number=}")
        routine = routines.get(routine)
        data.refreshFromFiles()
        success = routine(cal_number, data, attr="filtValue", calibratedName="energy")
        if success:
            self.calibration_state = "calibrated"
        elif self.calibration_state == "calibrated":
            self.calibration_state = "no_calibration"
        return self.calibration_state

    # Analysis Functions
    def _get_data(self):
        assert self._off_filename is not None, "self.off_filename is None"
        if self._data is None:
            self._data = ChannelGroup(getOffFileListFromOneFile(self._off_filename))
        self._data.refreshFromFiles()
        return self._data

    def roi_get_counts(self):
        """return a list of counts in each ROI for the last scan epoch time.
        Should call after scan_point_end"""
        # Need to put in TFY-XAS
        last_epoch_idx = len(self._scan.epoch_time_end_s) - 1
        start_unixnano = int(1e9)*self._scan.epoch_time_start_s[last_epoch_idx]
        end_unixnano = int(1e9)*self._scan.epoch_time_end_s[last_epoch_idx]
        # roi should always have at least TFY in it
        roi_counts = {}
        if self.calibration_state == "no_calibration":
            # check reasonable range for filtValue, or better yet, stop trying to histogram
            bin_centers, counts = self._get_data().histWithUnixnanos([10, 20000], "filtValue", [start_unixnano], [end_unixnano])
            roi_counts['tfy'] = int(counts[0])
        else:
            for name, (lo_ev, hi_ev) in self.roi.items():
                bin_centers, counts = self._get_data().histWithUnixnanos([lo_ev, hi_ev], "energy", [start_unixnano], [end_unixnano])
                roi_counts[name] = int(counts[0])
            #print(bin_centers, counts)
        return roi_counts

    def roi_save_counts(self):
        roi_counts = self.roi_get_counts()
        output_dir = self._beamtime_user_output_dir("pfy")
        output_file = os.path.join(output_dir, f"scan_{self._scan.scan_num}")
        roi_names = roi_counts.keys()
        data = np.array([roi_counts[name] for name in roi_names])
        header = " ".join(roi_names)
        if not os.path.isfile(output_file):
            print("ROI Save Counts", header)
            with open(output_file, "w") as f:
                np.savetxt(f, data[np.newaxis, :], header=header)
        else:
            with open(output_file, "a") as f:
                np.savetxt(f, data[np.newaxis, :])
        return roi_counts        
        
    def quick_post_process(self):
        """
        Will be broken right now due to ROI changes
        """
        assert self._last_scan.roi != {}, "rois_bin_edges is None: first call set_rois, then start_rois_counts, roi_start_counts, then roi_get_counts"
        #a, b = self.roi_counts_start_unixnano, time_unixnano()
        #self.roi_counts_start_unixnano = None
        if self.calibration_state == "no_calibration":
            return 
        data = self._get_data()
        roi_names = self._last_scan.roi.keys()
        attr = 'energy'
        all_roi_data = []
        for r in roi_names:
            bin_edges = self._last_scan.roi[r]
            hist2dres = self._last_scan.hist2d(data, bin_edges, attr)
            hist2d = hist2dres.hist2d
            all_roi_data.append(hist2d)
        roi_counts = np.vstack(all_roi_data)
        output_dir = self._beamtime_user_output_dir("quick_post_process")
        output_file = os.path.join(output_dir, f"scan_{self._last_scan.scan_num}")
        header = " ".join(roi_names)
        np.savetxt(output_file, roi_counts.T, header=header, fmt="%d")
        
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
    
    def _beamtime_user_output_dir(self, subdir = None, make = True):
        dirname = os.path.join(self.base_user_output_dir, f"beamtime_{self.beamtime_id}")
        if subdir is not None:
            dirname = os.path.join(dirname, subdir)
        if make:
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname    
    
    def _get_current_scan_num_from_logs(self):
        log_dir = self._user_log_dir(make=False)
        if exists(log_dir):
            scans = glob(join(log_dir, "scan*.json"))
            cals = glob(join(log_dir, "calibration*.json"))
            log_names = scans + cals
            nums = []
            for name in log_names:
                try:
                    nums.append(int(name[-9:-5]))
                except ValueError:
                    pass
            if nums == []:
                scan_num = 1
            else:
                scan_num = max(nums) + 1
        else:
            scan_num = 1
        return scan_num
            
    def _user_log_dir(self, make=True):
        return self._beamtime_user_output_dir("logs", make=make)

    def _tes_log_dir(self, make=True):
        dirname = os.path.join(os.path.dirname(self._off_filename),"logs")
        if make:
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

    def _log_filenames(self, log_name, log_num):
        # we duplicate logs
        # one set goes to the beamtime directory for user consumption
        # another set lives with the off files for convenience
        assert log_name in ["scan", "calibration"]
        filename1 = os.path.join(self._user_log_dir(), f"{log_name}{log_num:04d}.json")
        filename2 = os.path.join(self._tes_log_dir(), f"{log_name}{log_num:04d}.json")
        if not self.overwrite:
            assert not os.path.isfile(filename1), f"{filename1} already exists"
            assert not os.path.isfile(filename2), f"{filename2} already exists"
        return [filename1, filename2]        

    def _validate_drift_correction_plan(self, drift_correction_plan):
        if drift_correction_plan not in ["none", "basic", "before_and_after_cals"]:
            raise Exception("invalid drift plan")

    def _scan_user_output_dir(self, scan_num, subdir = None, make = True):
        dirname = self._beamtime_user_output_dir(f"scan{scan_num:04d}", make = make)
        if subdir is not None:
            dirname = os.path.join(dirname, subdir)
        if make:
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

def time_unixnano():
    return int(1e9*time.time())

