from scan_server import TESScanner, Scan, DastardClient, CalibrationLog, post_process
import pytest
import statemachine
import numpy as np
import tempfile
import os
from . import util
from pathlib import Path


class MockClient(DastardClient):
    expected_states = ["CAL0", "PAUSE", "SCAN0", "PAUSE", "SCAN1", "PAUSE", "CAL1", "PAUSE"]

    def _call(self, method, params):
        self._last_method = method
        self._last_params = params
        return None

    def _connect(self):
        return  # dont try to connect the socket

    def set_experiment_state(self, state):
        try:
            self.state_i
        except:
            self.state_i = 0
        assert isinstance(state, str)
        expected = self.expected_states[self.state_i]
        assert state == expected, f"got {state} for i={self.state_i}, expected {expected}"
        self.state_i += 1


class MockListener():
    def set_next(self, topic, contents):
        self.topic = topic
        self.contents = contents
    
    def get_message_with_topic(self, topic):
        assert isinstance(topic, str)
        assert topic == self.topic
        self.topic = None
        return self.contents

@pytest.mark.dependency()
def test_tes_scanner():
    base_user_output_dir = os.path.join(util.ssrl_dir, "base_user_output_dir")
    Path(base_user_output_dir).mkdir(parents=False, exist_ok=True)
    listener = MockListener()
    client = MockClient(("test_url", "test_port"), listener)
    bg_log_file = open(os.path.join(util.ssrl_dir, "background_process_log_file"), "w")
    scanner = TESScanner(dastard = client, beamtime_id ="test", base_user_output_dir=base_user_output_dir,
        background_process_log_file=bg_log_file)
    listener.set_next(topic="WRITING", contents ={"Active":True, "FilenamePattern": util.ssrl_filename_pattern})
    util.write_ssrl_experiment_state_file(util.ssrl_filename_pattern%("experiment_state","txt"))
    #write the full experiment state file all at once, it is much easier than emulating the data arriving
    scanner.file_start(True, False, None)
    assert client._last_method == "SourceControl.WriteControl"
    scanner.calibration_data_start(sample_id = 0, sample_desc = "test_sample", routine = "ssrl_10_1_mix_cal")
    scanner.calibration_data_end(_try_post_processing=False)
    scanner.calibration_learn_from_last_data()
    scanner.scan_start(var_name="mono", var_unit="eV", scan_num=0,  
                sample_id=0, sample_desc="test_desc", extra = {"tempK":43.2}, 
                drift_correction_plan = "before_and_after_cals")
    with pytest.raises(statemachine.exceptions.TransitionNotAllowed):
        scanner.file_start(True, False, None) # start file while file started not allowed
    scanner.scan_point_start(122, extra={})
    with pytest.raises(statemachine.exceptions.TransitionNotAllowed):
        scanner.scan_point_start(122, extra={}) # start point 2x in a row not allowed
    scanner.scan_point_end()
    scanner.scan_point_start(123, extra={})
    scanner.scan_point_end()
    scanner.scan_end(_try_post_processing=False)
    scanner.scan_start(var_name="mono", var_unit="eV", scan_num=1, 
                sample_id=0, sample_desc="test_desc", extra = {"another_var":"some text"}, 
                drift_correction_plan = "basic")
    scan_to_copy = util.scans()[1]
    for var_value, a, b in zip(scan_to_copy.var_values, scan_to_copy.epoch_time_start_s, scan_to_copy.epoch_time_end_s):
        scanner.scan_point_start(var_value, extra={}, _epoch_time_s_for_test=a)
        scanner.scan_point_end(_epoch_time_s_for_test=b)
    scanner.scan_end(_try_post_processing=False)
    # reach into internals to make the times correct, dont do this in real usage
    scanner.last_scan.epoch_time_end_s = scan_to_copy.epoch_time_end_s
    scanner.last_scan.epoch_time_start_s = scan_to_copy.epoch_time_start_s
    # done reaching into internals
    with pytest.raises(AssertionError):
        scanner.roi_set([(100, 150), (5, 550)]) # 2nd bin starts below first bin
    scanner.roi_set([(240, 300), (500, 550), (750, 800)])
    scanner.roi_start_counts()
    scanner.roi_get_counts()
    scanner.calibration_data_start(sample_id = 0, sample_desc = "test_sample", routine = "ssrl_10_1_mix_cal")
    scanner.calibration_data_end(_try_post_processing=False)
    # scanner.scan_start_calc_last_outputs()
    result = scanner._get_data().linefit("OKAlpha", attr="energy", plot=False)
    assert result.params["fwhm"].value < 7
    scanner.file_end()
    with pytest.raises(statemachine.exceptions.TransitionNotAllowed):
        scanner.file_end() # end file while file ended not allowed
 
    with open(os.path.join(util.ssrl_dir, "logs", "calibration0000.json"), "r") as f:
        cal0 = CalibrationLog.from_json(f.read())
    with open(os.path.join(util.ssrl_dir, "logs", "calibration0001.json"), "r") as f:
        cal1 = CalibrationLog.from_json(f.read())
    assert cal1 != cal0

    with open(os.path.join(base_user_output_dir, "beamtime_test", "logs", "scan0000.json"), "r") as f:
        scan = Scan.from_json(f.read())
    assert scan._ended
    assert scan.previous_cal_log == cal0


    # in normal use would be started by ending scans and calibrations
    # here we want to use fewer channels for speed, so we start it manually
    r = scanner.start_post_processing(_max_channels=3)
    assert r == "started new process"
    r = scanner.start_post_processing(_max_channels=3)
    assert r == "previous process still running"
    scanner.background_process.wait(timeout=30)

    assert os.path.isfile(os.path.join(base_user_output_dir, "beamtime_test/scan0001/scan0001_hist2d.png"))

def test_scan():
    scan = Scan(var_name="mono", var_unit="eV", scan_num=0, beamtime_id="test_Beamtime", 
                sample_id=0, sample_desc="test_desc", extra={}, data_path="no actual data", user_output_dir="dummy",
                previous_cal_log = CalibrationLog(1, 1, "", 1, "", "", "", 1), drift_correction_plan = "none")
    for i, mono_val in enumerate(np.arange(5)):
        start, end = i, i+0.5
        scan.point_start(mono_val, start, extra={})
        scan.point_end(end)
    scan.end()
    filename = tempfile.mktemp()
    d = scan.to_dict()
    scan2 = Scan.from_dict(d)
    assert scan == scan2
    scan.to_disk(filename)
    scan2 = Scan.from_file(filename)
    assert scan == scan2
    scan.experiment_state_file_as_str(header=True)
    assert len(str(scan)) > 10

@pytest.mark.dependency(depends=["test_tes_scanner"])
def test_post_process_pieces():
    beamtime_dir = os.path.join(util.ssrl_dir, "base_user_output_dir", "beamtime_test")
    scans, calibrations = post_process.get_scans_and_calibrations(beamtime_dir)
    # test_test_scanner creates user output, which we don't want to mess with
    # so to test logic about if we should process, we will change user_output_dir 
    # a temporary directory
    for scan in scans.values():
        scan.user_output_dir = tempfile.mktemp()
    assert len(scans) == 2
    assert len(calibrations) == 2

    assert not post_process.output_exists(scans[0])
    assert not post_process.output_exists(scans[1])

    Path(scans[0].user_output_dir).mkdir(parents=False, exist_ok=False)
    assert post_process.output_exists(scans[0])
    assert not post_process.should_process(scans[0], calibrations)
    assert post_process.should_process(scans[1], calibrations)


    scans_to_process = post_process.get_scans_to_process(scans, calibrations)
    assert 0 not in scans_to_process.keys()
    assert 1 in scans_to_process.keys()

    for scan in scans_to_process.values():
        scan_hist2d = post_process.process(scan, calibrations, 
            scan.drift_correction_plan, np.arange(0, 1000, 1), 
            scan.user_output_dir, max_channels = 3)

    assert scan_hist2d.hist2d.sum() == 1480

