from .scan_json import DataScan, CalibrationScan
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
import datetime
from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed
import subprocess
import os
from os.path import join, exists, basename, dirname
from pathlib import Path
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import time
from glob import glob
from .dastard_client import DastardError

# from .cringe_model import CringeControl
# from .adr_model import ADRListener
# from .cringe_model import CringeControl



@dataclass_json
@dataclass
class CringeDastardSettings:
    record_nsamples: int
    record_npresamples: int
    trigger_threshold: int
    trigger_n_monotonic: int
    write_off: bool
    write_ljh: bool
    projector_filename: str

    
class ScannerState(StateMachine):
    """defines allowed state transitions, transitions will error if you do an invalid one"""
    no_file = State('no_file', initial=True)
    file_open = State("file_open")
    scan = State('scan')
    scan_point = State("scan_point")
    # cal_data = State('cal_data')

    scan_start = file_open.to(scan)
    scan_end = scan.to(file_open)

    scan_point_start = scan.to(scan_point)
    scan_point_end = scan_point.to(scan)

    file_start = no_file.to(file_open)
    file_end = file_open.to(no_file)

    def __init__(self, signal):
        self.signal = signal
        super().__init__()
    
    def on_enter_state(self, state):
        self.signal.emit(state.name)

class TESModel(QObject):
    autotuned = pyqtSignal(str)
    crate_powered_on = pyqtSignal(str)
    crate_powered_off = pyqtSignal(str)
    programs_started = pyqtSignal(bool)
    programs_killed = pyqtSignal(bool)
    abaco_on = pyqtSignal(bool)
    abaco_off = pyqtSignal(bool)
    state_changed = pyqtSignal(str)
    autosetup_changed = pyqtSignal(bool)
    
    def __init__(self, dastard, beamtime_id: str, base_user_output_dir: str,
                 background_process_log_file, cdsettings):
        super().__init__()
        self._command_list = ['state', 'filename', 'scan_str', 'scan_num', 'next_scan_num',
                              'cal_number', 'getFilenamePattern', 'start_abaco', 'start_programs',
                              'kill_programs', 'check_programs_running', 'stop_abaco',
                              'file_start', 'file_end', 'make_projectors', 'set_projectors',
                              'set_pulse_triggers', 'set_noise_triggers', 'scan_start',
                              'scan_point_start', 'scan_point_end', 'calibration_start',
                              'scan_end', 'rsync_data', 'setup_tes']
        self._dastard = dastard


        # self._start_adr_listener()
        # self._cc = CringeControl()

        self._cdsettings = cdsettings
        self._base_user_output_dir = base_user_output_dir
        self._beamtime_id = beamtime_id
        self._background_process_log_file = background_process_log_file
        self._state: ScannerState = ScannerState(self.state_changed)
        self._autosetup = False
        self._reset()

    """
    def _start_adr_listener(self):
        self._adrListener = ADRListener()
        self._adrListener.event.connect(self.adr_event_handler)
        self._adrListener.start()
    """
    
    def _reset(self):
        self._last_scan = None
        self._log_date = datetime.datetime.today().strftime("%Y%m%2d")
        self._scan = None
        self._cal_number: int = -1
        self._scan_num = None
        self._scan_str = ""
        self._overwrite = False
        self._off_filename = None

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

    @property
    def autosetup(self):
        return self._autosetup

    @autosetup.setter
    def autosetup(self, should_autosetup):
        self.autosetup_changed.emit(should_autosetup)
        self._autosetup = should_autosetup
        
    def getFilenamePattern(self, path):
        """
        Bad name: really takes a path pattern (filled with strftime) where raw data is stored,
        and generates a filename pattern for the OFF files AND creates all required directories
        on the path to that filename
        path : /nsls2/data/sst/legacy/ucal/raw/%Y/%m/%2d (I think this is now /data -- not writing
        directly to Lustre anymore due to weird problems)
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

    """
    def adr_event_handler(self, event):
        print(event)
        if event == 'regulate_after_cycle':
            if self.autosetup:
                print("Autosetup TES after Cycle End")
                self.setup_tes()
            elif event == "start_mag_cycle":
                print("Trying to stop file writing, if necessary")
                try:
                    self.file_end()
                except (TransitionNotAllowed, DastardError):
                    pass
    """
    
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
        success = self.start_abaco(restart=True)
        print("starting abaco")
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

    def start_abaco(self, restart=False):
        source, running = self._dastard.get_source_status()
        if source.lower() == 'abaco' and running:
            print("abaco already running")
            if restart:
                self._dastard.stop_source()
                success = self._dastard.start_abaco()
            else:
                success = True
        else:
            print(source, running)
            success = self._dastard.start_abaco()
        self.abaco_on.emit(success)
        return success

    def stop_abaco(self):
        success = self._dastard.stop_source()
        self.abaco_off.emit(success)
        return success
    
    def start_programs(self, restart=False):
        if restart:
            print("killing programs first")
            self.kill_programs()
            time.sleep(2)
        subprocess.Popen(['open_tes_programs.sh'])
        time.sleep(5)
        success = self.check_programs_running()
        self.programs_started.emit(success)
        return success

    def kill_programs(self):
        subprocess.Popen(['close_tes_programs.sh'])
        self._dastard.listener.reset()
        
    def check_programs_running(self):
        programs = ["dastard", "dcom"]
        proc_returns = [subprocess.run(["pgrep", prog], stdout=subprocess.PIPE)
                        for prog in programs]
        for r, prog in zip(proc_returns, programs):
            if r.returncode == 1:
                return False
        return True


    # def power_on_tes(self):
    #     return self._cc.setup_crate()

    # def autotune(self):
    #     return self._cc.full_tune()

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

    def file_end(self, _try_rsync_data=False, **rsync_kwargs):
        self._state.file_end()
        self._dastard.stop_writing()
        if _try_rsync_data:
            self.rsync_data(**rsync_kwargs)
        self._reset()

    def make_projectors(self, noise_file, pulse_file):
        args = ["make_projectors", "-rio", self._cdsettings.projector_filename,
                pulse_file, noise_file]
        print(args)
        subprocess.run(args, stdout=self._background_process_log_file,
                       stderr=subprocess.STDOUT)

    def set_projectors(self, projector_filename=None):
        if projector_filename is None:
            projector_filename = self._cdsettings.projector_filename
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
        for fname in self._log_filenames("scan", self.scan_num):
            if not self._overwrite:
                assert not os.path.isfile(fname)
        data_path = self._dastard.get_data_path()
        self._scan = DataScan(var_name, var_unit, self.scan_num, self._beamtime_id,
                              sample_id, sample_desc, extra, data_path,
                              cal_number=self._cal_number)
        self._scan_str = f"SCAN{self.scan_num}"
        self._dastard.set_experiment_state(self.scan_str)

    def calibration_start(self, var_name: str, var_unit: str, sample_id: int,
                          sample_desc: str, extra: dict = {}):
        """
        start taking calibration data, ensure the appropriate x-rays are
        incident on the detector
        sample_id: int - for your reference
        sample_desc: str - for your reference
        routine: str - which function is used to generate calibration
        curves from the data
        """
        self._state.scan_start()
        # self.set_pulse_triggers()

        data_path = self._dastard.get_data_path()
        self._scan = CalibrationScan(var_name, var_unit, self.scan_num,
                                     self._beamtime_id, sample_id,
                                     sample_desc, extra, data_path)
        self._scan_str = f"CAL{self.scan_num}"
        self._dastard.set_experiment_state(self.scan_str)
        self._cal_number = self.scan_num

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
            # self.start_post_processing()
        if _try_rsync_data:
            self.rsync_data(**rsync_kwargs)

    def rsync_data(self, dest="/nsls2/data/sst/legacy/ucal/raw/%Y/%m/%2d", filename=None):
        if filename is None:
            filename = self._off_filename
        from_dir = dirname(filename)
        date = datetime.datetime.strptime(basename(dirname(from_dir)), "%Y%m%d")
        to_dir = datetime.datetime.strftime(date, dest)
        args = ["rsync", "-vrt", "--append", from_dir, to_dir]
        print(args)
        subprocess.Popen(args)

    # Below section is entirely concerned with creating/finding log filenames
    # Somehow, this should be locked away in a deep dark dungeon
    def _beamtime_user_output_dir(self, subdir=None, make=True):
        dirname = os.path.join(self._base_user_output_dir, f"beamtime_{self._beamtime_id}")
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
        #return 0

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
        print(f"log file names:\n{filename1=}\n{filename2=}")
        return [filename1, filename2]
