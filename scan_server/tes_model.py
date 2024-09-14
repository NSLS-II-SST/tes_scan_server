from .scan_json import DataScan, CalibrationScan
from PyQt5.QtCore import QObject, pyqtSignal
import datetime
import subprocess
import os
from os.path import join, exists, basename, dirname, expanduser
from pathlib import Path
import time
from .rpc_server import time_human
from glob import glob
from .dastard_client import DastardError
from shutil import copy


"""def signal_property(signal_type, default=None):
    def decorator(func):
        name = func.__name__
        private_name = f"_{name}"
        signal_name = f"{name}_changed"

        def getter(self):
            if not hasattr(self, private_name):
                setattr(self, private_name, default)
            return getattr(self, private_name)

        def setter(self, value):
            if not hasattr(self, private_name):
                setattr(self, private_name, default)
            if getattr(self, private_name) != value:
                setattr(self, private_name, value)
                getattr(self, signal_name).emit(value)

        return property(getter, setter)

    return decorator"""


# Commented out SignalProperty code


class SignalProperty:
    def __init__(self, default_value):
        self.value = default_value
        self.signal = None

    def __set_name__(self, obj, name):
        self.name = name
        self.signal_name = f"{name}_changed"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.value

    def __set__(self, obj, value):
        if self.value != value:
            self.value = value
            signal = getattr(obj, self.signal_name, None)

            if signal is not None:
                signal.emit(value)


"""
def setup_signals(cls):
    items = list(cls.__dict__.items())
    for name, attr in items:
        if isinstance(attr, SignalProperty):
            signal = pyqtSignal(object)
            setattr(cls, f"{name}_changed", signal)
    return cls
"""


class TESModel(QObject):
    # Signals that don't follow the SignalProperty pattern
    autotuned = pyqtSignal(str)
    crate_powered_on = pyqtSignal(str)
    crate_powered_off = pyqtSignal(str)
    programs_started = pyqtSignal(bool)
    programs_killed = pyqtSignal(bool)
    source_on = pyqtSignal(bool)
    source_off = pyqtSignal(bool)
    status_updated = pyqtSignal(dict)

    # Signals that follow the SignalProperty pattern
    filename_changed = pyqtSignal(str)
    noise_file_changed = pyqtSignal(str)
    projector_file_changed = pyqtSignal(str)
    dastard_connected_changed = pyqtSignal(bool)
    state_changed = pyqtSignal(str)
    autosetup_changed = pyqtSignal(bool)
    noise_uid_changed = pyqtSignal(str)
    projector_uid_changed = pyqtSignal(str)
    calibration_uid_changed = pyqtSignal(str)
    rsync_on_file_end_changed = pyqtSignal(bool)
    rsync_on_scan_end_changed = pyqtSignal(bool)
    write_ljh_changed = pyqtSignal(bool)
    write_off_changed = pyqtSignal(bool)
    scan_str_changed = pyqtSignal(str)
    scan_num_changed = pyqtSignal(int)
    writing_changed = pyqtSignal(bool)
    channel_names_changed = pyqtSignal(list)
    source_changed = pyqtSignal(str)
    running_changed = pyqtSignal(bool)
    projectors_changed = pyqtSignal(bool)

    # SignalProperty attributes
    filename = SignalProperty("")
    noise_file = SignalProperty("")
    projector_file = SignalProperty("")
    dastard_connected = SignalProperty(False)
    state = SignalProperty("no_file")
    autosetup = SignalProperty(False)
    noise_uid = SignalProperty("")
    projector_uid = SignalProperty("")
    calibration_uid = SignalProperty("")
    rsync_on_file_end = SignalProperty(False)
    rsync_on_scan_end = SignalProperty(False)
    write_ljh = SignalProperty(True)
    write_off = SignalProperty(False)
    scan_str = SignalProperty("")
    scan_num = SignalProperty(0)
    writing = SignalProperty(False)
    channel_names = SignalProperty([])
    source = SignalProperty("none")
    running = SignalProperty(False)
    projectors = SignalProperty(False)

    def __init__(self, dastard, config, adr=None, cringe=None):
        super().__init__()
        self._dastard = dastard
        self._config = config
        self._cc = cringe
        self._adrListener = adr
        self._load_config()
        if self._adrListener is not None:
            self._start_adr_listener()

        self._reset()

        # Connect DastardClient signals
        self.connect_to_attribute(self._dastard.writing_changed, "writing")
        self.connect_to_attribute(self._dastard.channel_names_changed, "channel_names")
        self.connect_to_attribute(self._dastard.source_changed, "source")
        self.connect_to_attribute(self._dastard.running_changed, "running")
        self.connect_to_attribute(self._dastard.connected_changed, "dastard_connected")
        self.connect_to_attribute(self._dastard.projectors_changed, "projectors")

        self._dastard.status_updated.connect(self.status_updated)
        self._dastard.filename_changed.connect(self.handle_filename_changed)

    def connect_to_attribute(self, signal, attribute_name):
        signal.connect(lambda x: setattr(self, attribute_name, x))

    def handle_filename_changed(self, filename):
        if self.filename != filename:
            self.filename = filename
            if filename != "":
                self._log_date = os.path.basename(filename)[:8]
                self.state = "file_open"
            else:
                self.state = "no_file"

    def _start_adr_listener(self):
        self._adrListener.event.connect(self.adr_event_handler)
        self._adrListener.start()

    def _load_config(self):
        self._base_user_output_dir = expanduser(
            self._config.get("base_user_output_dir")
        )
        self._beamtime_id = self._config.get("beamtime_id")
        server_log_dir = expanduser(self._config.get("server_log_dir"))
        Path(server_log_dir).mkdir(parents=True, exist_ok=True)

        bg_log_file = open(os.path.join(server_log_dir, f"{time_human()}_bg.log"), "a")
        self._background_process_log_file = bg_log_file

    def _reset(self):
        self.noise_uid = ""
        self.projector_uid = ""
        self.calibration_uid = ""
        self.noise_file = ""
        self.projector_file = ""
        self.rsync_on_file_end = self._config.get("rsync_on_file_end", False)
        self.rsync_on_scan_end = self._config.get("rsync_on_scan_end", False)
        self._last_scan = None
        self._log_date = datetime.datetime.today().strftime("%Y%m%2d")
        self._scan = None
        self._cal_number = -1
        self.scan_num = self._get_current_scan_num_from_logs()
        self.scan_str = ""
        self._overwrite = False
        self.filename = ""
        self._last_projector_file = None
        self.write_ljh = self._config.get("write_ljh", True)
        self.write_off = False
        self.writing = False
        self.channel_names = []
        self.source = "none"
        self.running = self._dastard.running

    @property
    def next_scan_num(self):
        return self.scan_num + 1

    @property
    def cal_number(self):
        return self._cal_number

    def _advance_scan_num(self):
        self.scan_num = self.scan_num + 1
        return self.scan_num

    def getFilenamePattern(self, path):
        today = datetime.datetime.today()
        datedir = today.strftime(path)
        """for i in range(1000):
            sampledir = join(datedir, f"{i:04d}")
            if not exists(sampledir):
                # os.makedirs(sampledir)
                filepattern = join(
                    sampledir, today.strftime(f"%Y%m%2d_run{i:04d}_%%s.%%s")
                )
                return filepattern
        """
        #raise ValueError("Could not find a suitable directory name")
        return datedir
    

    def adr_event_handler(self, event):
        print(event)
        if event == "regulate_after_cycle":
            if self.autosetup:
                print("Autosetup TES after Cycle End")
                self.setup_tes()
            elif event == "start_mag_cycle":
                print("Trying to stop file writing, if necessary")
                try:
                    self.file_end()
                except DastardError:
                    pass

    def setup_tes(self):
        print("starting programs")
        success = self.start_programs(restart=True)
        if not success:
            print("failure")
            return success
        print("success")
        print("powering TES")
        success = self.power_on_tes()
        if not success:
            print("failure")
            return success
        print("success")
        success = self.start_source(restart=True)
        print("starting source")
        if not success:
            print("failure")
            return success
        print("success")
        print("starting autotune")
        success = self.autotune()
        if success:
            print("success")
        else:
            print("failure")
        return success

    # Dastard operations
    def start_source(self, restart=False):
        success = self._dastard.start_source(restart=False)
        self.source_on.emit(success)
        return success

    def stop_source(self):
        success = self._dastard.stop_source()
        self.source_off.emit(success)
        return success

    def start_programs(self, restart=False):
        programs = self._config.get("programs_to_run", [])

        if restart:
            print("killing programs first")
            self.kill_programs()
            time.sleep(2)
        args = ["open_tes_programs.sh"] + programs
        print("Running ", args)
        subprocess.Popen(args)
        time.sleep(5)
        success = self.check_programs_running()
        self.programs_started.emit(success)
        return success

    def kill_programs(self):
        subprocess.Popen(["close_tes_programs.sh"])
        self._dastard.listener.reset()

    def check_programs_running(self):
        programs = self._config.get("programs_to_run", [])
        proc_returns = [
            subprocess.run(["pgrep", prog], stdout=subprocess.PIPE) for prog in programs
        ]
        for r, prog in zip(proc_returns, programs):
            if r.returncode == 1:
                return False
        return True

    def power_on_tes(self):
        if self._cc is not None:
            result = self._cc.setup_crate()
            self.crate_powered_on.emit(result)
        else:
            result = True
        return result

    def autotune(self):
        if self._cc is not None:
            self._cc.send_all_tower()
            self._cc.shock_db1()
            result = self._cc.full_tune()
            self.set_pulse_triggers()
            self.autotuned.emit(result)
            return result
        else:
            return True

    def file_start(
        self, path=None, write_ljh=None, write_off=None, setFilenamePattern=False
    ):
        """
        tell dastard to start a new file, must be called before any
        calibration or scan functions
        """
        if setFilenamePattern:
            filenamePattern = self.getFilenamePattern(path)
        else:
            filenamePattern = None
        try:
            self.filename = self._dastard.start_file(
                write_ljh if write_ljh is not None else self.write_ljh,
                write_off if write_off is not None else self.write_off,
                path,
                filenamePattern,
            )
            self._log_date = os.path.basename(self.filename)[:8]
            self.state = "file_open"
        except DastardError as e:
            self.filename = ""
            raise e
        return self.filename

    def file_end(self, _try_rsync_data=None, **rsync_kwargs):
        self.state = "no_file"
        self._dastard.stop_writing()
        if _try_rsync_data is None:
            _try_rsync_data = self.rsync_on_file_end
        if _try_rsync_data:
            self.rsync_data(**rsync_kwargs)
        # self._reset()

    def make_projectors(self, noise_file, pulse_file):
        projector_filename = expanduser(self._config["projector_filename"])
        args = []
        args += self._config.get("projector_cmd", ["make_projectors", "-ro"])
        args += [projector_filename, pulse_file, noise_file]

        pulse_folder = os.path.dirname(pulse_file)
        print(args)
        proc_return = subprocess.run(
            args, stdout=self._background_process_log_file, stderr=subprocess.STDOUT
        )
        if proc_return.returncode == 0:
            return True
        else:
            return False

        #copy(projector_filename, pulse_folder)

    def set_projectors(self, projector_filename=None):
        if projector_filename is None:
            projector_filename = expanduser(self._config.get("projector_filename"))
        self._dastard.set_projectors(projector_filename)
        self._write_off = self._config.get("write_off", True)
        self.write_off_changed.emit(self._write_off)

    def set_pulse_triggers(self):
        self._dastard.configure_record_lengths()
        self._dastard.zero_all_triggers()
        self._dastard.set_pulse_trigger_all_chans()

    def set_noise_triggers(self):
        self._dastard.configure_record_lengths()
        self._dastard.zero_all_triggers()
        self._dastard.set_noise_trigger_all_chans()

    # Scan operations
    def scan_start(
        self,
        var_name: str,
        var_unit: str,
        sample_id: int,
        sample_desc: str,
        extra: dict = {},
    ):
        if not self._dastard.writing:
            raise RuntimeError("No file is open!")
        for fname in self._log_filenames("scan", self.scan_num):
            if not self._overwrite:
                assert not os.path.isfile(fname)
        data_path = self._dastard.get_data_path()
        self._scan = DataScan(
            var_name,
            var_unit,
            self.scan_num,
            self._beamtime_id,
            sample_id,
            sample_desc,
            extra,
            data_path,
            cal_number=self._cal_number,
        )
        self.scan_str = f"SCAN{self.scan_num}"
        self._dastard.set_experiment_state(self.scan_str)
        self.state = "scan"

    def calibration_start(
        self,
        var_name: str,
        var_unit: str,
        sample_id: int,
        sample_desc: str,
        extra: dict = {},
    ):
        """
        start taking calibration data, ensure the appropriate x-rays are
        incident on the detector
        sample_id: int - for your reference
        sample_desc: str - for your reference
        routine: str - which function is used to generate calibration
        curves from the data
        """
        # self._state.scan_start()
        # self.set_pulse_triggers()
        if not self._dastard.writing:
            raise RuntimeError("No file is open!")
        data_path = self._dastard.get_data_path()
        self._scan = CalibrationScan(
            var_name,
            var_unit,
            self.scan_num,
            self._beamtime_id,
            sample_id,
            sample_desc,
            extra,
            data_path,
        )
        self.scan_str = f"CAL{self.scan_num}"
        self._dastard.set_experiment_state(self.scan_str)
        self._cal_number = self.scan_num
        self.state = "calibration"

    def scan_point_start(
        self, scan_var: float, _epoch_time_s_for_test=None, extra: dict = None
    ):
        if _epoch_time_s_for_test is None:
            _epoch_time_s_for_test = time.time()
        self._scan.point_start(scan_var, _epoch_time_s_for_test, extra)
        return _epoch_time_s_for_test

    def scan_point_end(self, _epoch_time_s_for_test=None):
        if _epoch_time_s_for_test is None:
            _epoch_time_s_for_test = time.time()
        self._scan.point_end(_epoch_time_s_for_test)
        return _epoch_time_s_for_test

    def scan_end(
        self, _try_post_processing=False, _try_rsync_data=None, **rsync_kwargs
    ):
        if self._scan is not None:
            self._scan.end()
            scan_name = "calibration" if self._scan.calibration else "scan"
            for fname in self._log_filenames(scan_name, self._scan.scan_num):
                self._scan.to_disk(fname, self._overwrite)
            self._last_scan = self._scan
            self._advance_scan_num()
            self._scan = None
            self.scan_str = ""
            self._dastard.set_experiment_state("PAUSE")
            self.state = "file_open"
            if _try_post_processing:
                pass
            if _try_rsync_data is None:
                _try_rsync_data = self.rsync_on_scan_end
            if _try_rsync_data:
                self.rsync_data(**rsync_kwargs)
        else:
            self.scan_str = ""
            self._dastard.set_experiment_state("PAUSE")
            return "No scan was open"

    def rsync_data(
        self, dest="/nsls2/data/sst/legacy/ucal/raw/%Y/%m/%2d", filename=None
    ):
        if filename is None:
            filename = self.filename
        if filename is None:
            print("No file given, not going to rsync")
            return

        if dest is None:
            dest = self._config.get("rsync_dest", None)
        if dest is None:
            print("Received no destination, not rsyncing anywhere")
            return

        from_dir = dirname(filename)
        date = datetime.datetime.strptime(basename(dirname(from_dir)), "%Y%m%d")
        to_dir = datetime.datetime.strftime(date, dest)
        args = ["rsync", "-vrt", "--append", from_dir, to_dir]
        print(args)
        subprocess.Popen(args)

    # Below section is entirely concerned with creating/finding log filenames
    # Somehow, this should be locked away in a deep dark dungeon
    def _beamtime_user_output_dir(self, subdir=None, make=True):
        dirname = os.path.join(
            self._base_user_output_dir, f"beamtime_{self._beamtime_id}"
        )
        if subdir is not None:
            dirname = os.path.join(dirname, subdir)
        if make:
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

    def _user_log_dir(self, make=True):
        return self._beamtime_user_output_dir(self._log_date, make=make)

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

    def _tes_log_dir(self, make=True):
        dirname = os.path.join(os.path.dirname(self.filename), "logs")
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
        #filename2 = os.path.join(self._tes_log_dir(), f"{log_name}{log_num:04d}.json")
        lognames = [filename1]
        if not self._overwrite:
            for filename in lognames:
                assert not os.path.isfile(filename), f"{filename} already exists"
        return lognames
