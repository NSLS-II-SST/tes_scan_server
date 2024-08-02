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


class TESModel(QObject):
    autotuned = pyqtSignal(str)
    crate_powered_on = pyqtSignal(str)
    crate_powered_off = pyqtSignal(str)
    programs_started = pyqtSignal(bool)
    programs_killed = pyqtSignal(bool)
    source_on = pyqtSignal(bool)
    source_off = pyqtSignal(bool)
    state_changed = pyqtSignal(str)
    autosetup_changed = pyqtSignal(bool)
    filename_changed = pyqtSignal(str)
    scan_str_changed = pyqtSignal(str)
    scan_num_changed = pyqtSignal(int)
    noise_uid_changed = pyqtSignal(str)
    projector_uid_changed = pyqtSignal(str)
    rsync_on_file_end_changed = pyqtSignal(bool)
    rsync_on_scan_end_changed = pyqtSignal(bool)
    write_ljh_changed = pyqtSignal(bool)
    write_off_changed = pyqtSignal(bool)

    # New signals to relay from DastardClient
    writing_changed = pyqtSignal(bool)
    channel_names_changed = pyqtSignal(list)
    status_updated = pyqtSignal(dict)
    source_changed = pyqtSignal(str)
    running_changed = pyqtSignal(bool)
    dastard_connected_changed = pyqtSignal(bool)

    def __init__(self, dastard, config, adr=None, cringe=None):
        super().__init__()

        self._dastard = dastard
        self._config = config
        self._cc = cringe
        self._adrListener = adr
        self._load_config()
        if self._adrListener is not None:
            self._start_adr_listener()

        self._state = "no_file"
        self._autosetup = False
        self._reset()

        # Connect DastardClient signals
        self._dastard.state_changed.connect(self.state_changed)
        self._dastard.writing_changed.connect(self.writing_changed)
        self._dastard.channel_names_changed.connect(self.channel_names_changed)
        self._dastard.status_updated.connect(self.status_updated)
        self._dastard.source_changed.connect(self.source_changed)
        self._dastard.running_changed.connect(self.running_changed)
        self._dastard.filename_changed.connect(self.handle_filename_changed)
        self._dastard.connected_changed.connect(self.handle_connected_changed)

    def handle_filename_changed(self, filename):
        if self._off_filename != filename:
            self._off_filename = filename
            self.filename_changed.emit(filename)
            if filename != "":
                self._log_date = os.path.basename(self._off_filename)[:8]
                self.state = "file_open"
            else:
                self.state = "no_file"

    def handle_connected_changed(self, connected):
        # Handle connection status changes here
        print(
            f"Dastard connection status changed: {'Connected' if connected else 'Disconnected'}"
        )
        self.dastard_connected_changed.emit(connected)

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
        self._noise_uid = ""
        self._projector_uid = ""
        self._rsync_on_file_end = self._config.get("rsync_on_file_end", False)
        self._rsync_on_scan_end = self._config.get("rsync_on_scan_end", False)
        self._last_scan = None
        self._log_date = datetime.datetime.today().strftime("%Y%m%2d")
        self._scan = None
        self._cal_number: int = -1
        self._scan_num = None
        self._scan_str = ""
        self._overwrite = False
        self._off_filename = None
        self._last_projector_file = None
        self._write_ljh = self._config.get("write_ljh", True)
        self._write_off = False

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        if self._state != new_state:
            self._state = new_state
            self.state_changed.emit(self._state)

    @property
    def filename(self):
        return self._off_filename

    @property
    def scan_str(self):
        return self._scan_str

    @scan_str.setter
    def scan_str(self, value):
        if self._scan_str != value:
            self._scan_str = value
            self.scan_str_changed.emit(self._scan_str)

    @property
    def scan_num(self):
        if self._scan_num is None:
            self._scan_num = self._get_current_scan_num_from_logs()
        return self._scan_num

    @scan_num.setter
    def scan_num(self, value):
        if self._scan_num != value:
            self._scan_num = value
            self.scan_num_changed.emit(self._scan_num)

    @property
    def next_scan_num(self):
        return self.scan_num + 1

    @property
    def cal_number(self):
        return self._cal_number

    def _advance_scan_num(self):
        self.scan_num = self.scan_num + 1
        return self._scan_num

    @property
    def autosetup(self):
        return self._autosetup

    @autosetup.setter
    def autosetup(self, should_autosetup):
        self.autosetup_changed.emit(should_autosetup)
        self._autosetup = should_autosetup

    @property
    def noise_uid(self):
        return self._noise_uid

    @noise_uid.setter
    def noise_uid(self, value):
        if self._noise_uid != value:
            self._noise_uid = value
            self.noise_uid_changed.emit(self._noise_uid)

    @property
    def projector_uid(self):
        return self._projector_uid

    @projector_uid.setter
    def projector_uid(self, value):
        if self._projector_uid != value:
            self._projector_uid = value
            self.projector_uid_changed.emit(self._projector_uid)

    @property
    def rsync_on_file_end(self):
        return self._rsync_on_file_end

    @rsync_on_file_end.setter
    def rsync_on_file_end(self, value):
        if self._rsync_on_file_end != value:
            self._rsync_on_file_end = value
            self.rsync_on_file_end_changed.emit(self._rsync_on_file_end)

    @property
    def rsync_on_scan_end(self):
        return self._rsync_on_scan_end

    @rsync_on_scan_end.setter
    def rsync_on_scan_end(self, value):
        if self._rsync_on_scan_end != value:
            self._rsync_on_scan_end = value
            self.rsync_on_scan_end_changed.emit(self._rsync_on_scan_end)

    @property
    def write_ljh(self):
        return self._write_ljh

    @write_ljh.setter
    def write_ljh(self, value):
        if self._write_ljh != value:
            self._write_ljh = value
            self.write_ljh_changed.emit(self._write_ljh)

    @property
    def write_off(self):
        return self._write_off

    @write_off.setter
    def write_off(self, value):
        if self._write_off != value:
            self._write_off = value
            self.write_off_changed.emit(self._write_off)

    def getFilenamePattern(self, path):
        today = datetime.datetime.today()
        datedir = today.strftime(path)
        for i in range(1000):
            sampledir = join(datedir, f"{i:04d}")
            if not exists(sampledir):
                os.makedirs(sampledir)
                filepattern = join(
                    sampledir, today.strftime(f"%Y%m%2d_run{i:04d}_%%s.%%s")
                )
                return filepattern
        raise ValueError("Could not find a suitable directory name")

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
        subprocess.Popen(["open_tes_programs.sh"] + programs)
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
            self._off_filename = self._dastard.start_file(
                write_ljh if write_ljh is not None else self.write_ljh,
                write_off if write_off is not None else self.write_off,
                path,
                filenamePattern,
            )
            self.filename_changed.emit(self._off_filename)

        except DastardError as e:
            self._off_filename = None
            raise e
        return self._off_filename

    def file_end(self, _try_rsync_data=None, **rsync_kwargs):
        self._dastard.stop_writing()
        if _try_rsync_data is None:
            _try_rsync_data = self.rsync_on_file_end
        if _try_rsync_data:
            self.rsync_data(**rsync_kwargs)
        self._reset()

    def make_projectors(self, noise_file, pulse_file):
        projector_filename = expanduser(self._config["projector_filename"])
        args = [
            "make_projectors",
            "-rio",
            projector_filename,
            pulse_file,
            noise_file,
        ]
        pulse_folder = os.path.dirname(pulse_file)
        print(args)
        subprocess.run(
            args, stdout=self._background_process_log_file, stderr=subprocess.STDOUT
        )
        copy(projector_filename, pulse_folder)

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
        if not self._dastard.is_writing():
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
        if not self._dastard.is_writing():
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
            filename = self._off_filename
        if filename is None:
            print("No file given, not going to rsync")
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
