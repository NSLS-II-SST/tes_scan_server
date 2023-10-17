from statemachine import StateMachine, State
from typing import List
import time
import datetime
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import io
import os
from os.path import join, exists, basename, dirname
from pathlib import Path
import subprocess
from glob import glob


@dataclass_json
@dataclass
class CringeDastardSettings:
    record_nsamples: int
    record_npresamples: int
    trigger_threshold: int
    trigger_n_monotonic: int
    write_off: bool
    write_ljh: bool


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
    user_output_dir: str
    point_extras: dict = field(default_factory=dict)
    var_values: List[float] = field(default_factory=list)
    epoch_time_start_s: List[int] = field(default_factory=list)
    epoch_time_end_s: List[int] = field(default_factory=list)
    _ended: bool = field(default=False)

    def point_start(self, scan_var, epoch_time_s, extra=None):
        assert not self._ended
        assert len(self.epoch_time_start_s) == len(self.epoch_time_end_s)

        self.var_values.append(float(scan_var))
        if extra is not None and extra != {}:
            assert isinstance(extra, dict)
            idx = str(len(self.epoch_time_start_s))
            self.point_extras[idx] = extra
        # print(f"{self.epoch_time_start_s}")
        self.epoch_time_start_s.append(epoch_time_s)
        # print(f"{self.epoch_time_start_s}")

    def point_end(self, epoch_time_s):
        assert len(self.epoch_time_start_s) - 1 == len(self.epoch_time_end_s)
        self.epoch_time_end_s.append(epoch_time_s)

    def write_experiment_state_file(self, f, header):
        if header:
            f.write("# unixnano, state label\n")
        for i, (start, end) in enumerate(zip(self.epoch_time_start_s, self.epoch_time_end_s)):
            label = f"SCAN{self.scan_num}_{i}"
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
    def __init__(self, dastard, beamtime_id: str, base_user_output_dir: str,
                 background_process_log_file, cdsettings):
        self._dastard = dastard
        self._beamtime_id = beamtime_id
        self._cdsettings = cdsettings
        self._base_user_output_dir = base_user_output_dir
        self._state: ScannerState = ScannerState()
        self._tfy_ulim = 1600
        self._tfy_llim = 200
        self._reset()
        self._background_process_log_file = background_process_log_file

    def _reset(self):
        self._last_scan = None
        self._log_date = datetime.datetime.today().strftime("%Y%m%2d")
        self._scan = None
        self._data = None
        self._cal_number: int = -1
        self._scan_num = None
        self._scan_str = ""
        self._overwrite = False
        self._calibration_to_routine: List[str] = []
        self.calibration_state = "no_calibration"
        self._off_filename = None

    # Properties
    @property
    def state(self):
        return self._state.current_state_value

    @property
    def filename(self):
        return self._off_filename

    @property
    def scan_str(self):
        return self._scan_str

    @property
    def scan_num(self):
        if self._scan_num is None:
            self._scan_num = self._get_current_scan_num_from_logs()
        return self._scan_num

    @scan_num.setter
    def scan_num(self, scan_num):
        self._scan_num = scan_num
        return self._scan_num

    @property
    def next_scan_num(self):
        return self.scan_num + 1

    @property
    def cal_number(self):
        return self._cal_number

    def _advance_scan_num(self):
        self._scan_num = self.scan_num + 1
        return self._scan_num

    def getFilenamePattern(self, path):
        """
        path : /nsls2/data/sst/legacy/ucal/raw/%Y/%m/%2d
        """
        today = datetime.datetime.today()
        datedir = today.strftime(path)
        for i in range(1000):
            sampledir = join(datedir, f"{i:04d}")
            if not exists(sampledir):
                os.makedirs(sampledir)
                filepattern = join(sampledir, today.strftime(f"%Y%m%2d_run{i:04d}_%%s.%%s"))
                return filepattern
        raise ValueError("Could not find a suitable directory name")

    def start_lancero(self):
        success = self._dastard.start_lancero()
        return success

    # Dastard operations
    def file_start(self, path=None, write_ljh=None, write_off=None,
                   setFilenamePattern=False):
        """
        tell dastard to start a new file, must be called before any
        calibration or scan functions
        """
        if write_ljh is None:
            write_ljh = self._cdsettings.write_ljh
        if write_off is None:
            write_off = self._cdsettings.write_off
        self._state.file_start()
        if setFilenamePattern:
            filenamePattern = self.getFilenamePattern(path)
        else:
            filenamePattern = None
        self._off_filename = self._dastard.start_file(write_ljh, write_off, path, filenamePattern)
        self._log_date = os.path.basename(self._off_filename)[:8]
        return self._off_filename
        # dastard lazily creates off files when it has data to write
        # so we need to wait to open the off files until some time has
        # passed from calling file_start
        # so we must always access through _get_data (_hides it from the rpc)

    def file_end(self, _try_rsync_data=False, **rsync_kwargs):
        self._state.file_end()
        self._dastard.stop_writing()
        if _try_rsync_data:
            self.rsync_data(**rsync_kwargs)
        self._reset()

    def make_projectors(self, noise_file, pulse_file):
        args = ["make_projectors", "-rio", "/home/xf07id1/.scan_server/nsls_projectors.hdf5", pulse_file, noise_file]
        print(args)
        subprocess.run(args, stdout=self._background_process_log_file, stderr=subprocess.STDOUT)

    def set_projectors(self, projector_filename="/home/xf07id1/.scan_server/nsls_projectors.hdf5"):
        self._dastard.set_projectors(projector_filename)

    def set_pulse_triggers(self):
        # ideally record length and the trigger settings would easily vary based on config
        # so they should live in nsls_server.py
        self._dastard.configure_record_lengths(nsamp=self._cdsettings.record_nsamples,
                                               npre=self._cdsettings.record_npresamples)
        self._dastard.zero_all_triggers()
        self._dastard.set_pulse_trigger_all_chans(threshold=self._cdsettings.trigger_threshold,
                                                  n_monotone=self._cdsettings.trigger_n_monotonic)

    def set_noise_triggers(self):
        self._dastard.configure_record_lengths(nsamp=self._cdsettings.record_nsamples,
                                               npre=self._cdsettings.record_npresamples)
        self._dastard.zero_all_triggers()
        self._dastard.set_noise_trigger_all_chans()

    # Scan operations
    def scan_start(self, var_name: str, var_unit: str, sample_id: int,
                   sample_desc: str, extra: dict = {}):
        self._state.scan_start()
        scan_num = self.scan_num
        for fname in self._log_filenames("scan", scan_num):
            if not self._overwrite:
                assert not os.path.isfile(fname)
        data_path = self._dastard.get_data_path()
        user_output_dir = self._scan_user_output_dir(scan_num)
        self._scan = DataScan(var_name, var_unit, scan_num, self._beamtime_id,
                              sample_id, sample_desc, extra, data_path,
                              cal_number=self._cal_number,
                              user_output_dir=user_output_dir)
        self._scan_str = f"SCAN{scan_num}"
        self._dastard.set_experiment_state(self.scan_str)

    def scan_point_start(self, scan_var: float, _epoch_time_s_for_test=None,
                         extra: dict = None):
        self._state.scan_point_start()
        if _epoch_time_s_for_test is None:
            _epoch_time_s_for_test = time.time()
        self._scan.point_start(scan_var, _epoch_time_s_for_test, extra)
        return _epoch_time_s_for_test

    def scan_point_end(self, _epoch_time_s_for_test=None):
        self._state.scan_point_end()
        if _epoch_time_s_for_test is None:
            _epoch_time_s_for_test = time.time()
        self._scan.point_end(_epoch_time_s_for_test)
        return _epoch_time_s_for_test

    def scan_end(self, _try_post_processing=False, _try_rsync_data=False, **rsync_kwargs):
        self._state.scan_end()
        self._scan.end()
        scan_name = "calibration" if self._scan.calibration else "scan"
        for fname in self._log_filenames(scan_name, self._scan.scan_num):
            self._scan.to_disk(fname, self._overwrite)
        self._last_scan = self._scan
        self._advance_scan_num()
        self._scan = None
        self._scan_str = ""
        self._dastard.set_experiment_state("PAUSE")
        if _try_post_processing:
            pass
            #self.start_post_processing()
        if _try_rsync_data:
            self.rsync_data(**rsync_kwargs)

    # Calibration Functions
    def calibration_start(self, var_name: str, var_unit: str, sample_id: int,
                          sample_desc: str, routine: str, extra: dict = {}):
        """
        start taking calibraion data, ensure the appropriate x-rays are
        incident on the detector
        sample_id: int - for your reference
        sample_desc: str - for your reference
        routine: str - which function is used to generate calibration
        curves from the data
        """
        self._state.scan_start()
        # self.set_pulse_triggers()

        self._calibration_to_routine.append(routine)
        data_path = self._dastard.get_data_path()
        scan_num = self.scan_num
        user_output_dir = self._scan_user_output_dir(scan_num)
        self._scan = CalibrationScan(var_name, var_unit, scan_num,
                                     self._beamtime_id, sample_id,
                                     sample_desc, extra, data_path,
                                     user_output_dir=user_output_dir)
        self._scan_str = f"CAL{scan_num}"
        self._dastard.set_experiment_state(self.scan_str)
        self._cal_number = scan_num

    # Analysis Functions

    def rsync_data(self, dest="/nsls2/data/sst/legacy/ucal/raw/%Y/%m/%2d", filename=None):
        if filename is None:
            filename = self._off_filename
        from_dir = dirname(filename)
        date = datetime.datetime.strptime(basename(dirname(from_dir)), "%Y%m%d")
        to_dir = datetime.datetime.strftime(date, dest)
        args = ["rsync", "-vrt", "--append", from_dir, to_dir]
        print(args)
        subprocess.Popen(args)

    def _beamtime_user_output_dir(self, subdir=None, make=True):
        dirname = os.path.join(self._base_user_output_dir, f"beamtime_{self._beamtime_id}")
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
                scan_num = 0
            else:
                scan_num = max(nums) + 1
        else:
            scan_num = 0
        return scan_num
        #return 0

    def _user_log_dir(self, make=True):
        return self._beamtime_user_output_dir(self._log_date, make=make)

    def _tes_log_dir(self, make=True):
        dirname = os.path.join(os.path.dirname(self._off_filename), "logs")
        if make:
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

    def _log_filenames(self, log_name, log_num):
        # we duplicate logs
        # one set goes to the beamtime directory for user consumption
        # another set lives with the off files for convenience
        assert log_name in ["scan", "calibration"]
        # user log is supposed to provide one stop shop to get an overview of all the data taken
        filename1 = os.path.join(self._user_log_dir(), f"{log_name}{log_num:04d}.json")
        # tes_log_dir lives right inside the ljh/off folder
        filename2 = os.path.join(self._tes_log_dir(), f"{log_name}{log_num:04d}.json")
        if not self._overwrite:
            assert not os.path.isfile(filename1), f"{filename1} already exists"
            assert not os.path.isfile(filename2), f"{filename2} already exists"
        return [filename1, filename2]

    def _scan_user_output_dir(self, scan_num, subdir=None, make=False):
        dirname = self._beamtime_user_output_dir(os.path.join(self._log_date,
                                                              f"scan{scan_num:04d}"),
                                                 make=make)
        if subdir is not None:
            dirname = os.path.join(dirname, subdir)
        if make:
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname


def time_unixnano():
    return int(1e9*time.time())
