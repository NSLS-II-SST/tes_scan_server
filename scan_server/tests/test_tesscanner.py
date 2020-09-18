from scan_server import TESScanner, Scan, DastardClient
import pytest
import statemachine
import numpy as np
import tempfile
import os

class MockClient(DastardClient):
    def _call(self, method, params):
        self._last_method = method
        self._last_params = params
        return None

    def _connect(self):
        return  # dont try to connect the socket


class MockListener():
    def set_next(self, topic, contents):
        self.topic = topic
        self.contents = contents
    
    def get_message_with_topic(self, topic):
        assert isinstance(topic, str)
        assert topic == self.topic
        self.topic = None
        return self.contents

def test_tes_scanner():
    base_log_dir = tempfile.TemporaryDirectory().name 
    listener = MockListener()
    client = MockClient(("test_url", "test_port"), listener, pulse_trigger_params = None, noise_trigger_params = None)
    scanner = TESScanner(dastard = client, beamtime_id ="test", base_log_dir=base_log_dir)
    listener.set_next(topic="WRITING", contents ={"Active":True, "FilenamePattern": "test_pattern"})
    scanner.file_start()
    assert client._last_method == "SourceControl.WriteControl"
    scanner.calibration_data_start(sample_id = 0, sample_name = "test_sample", routine = "ssrl_10_1_mix_cal")
    scanner.calibration_data_end()
    scanner.calibration_learn_from_last_data()
    scanner.scan_define(var_name="mono", var_unit="eV", scan_num=0, beamtime_id="test_scan", 
                ext_id=0, sample_id=0, sample_desc="test_desc", extra = {"tempK":43.2})
    with pytest.raises(statemachine.exceptions.TransitionNotAllowed):
        scanner.file_start() # start file while file started not allowed
    scanner.scan_point_start(122, extra={})
    with pytest.raises(statemachine.exceptions.TransitionNotAllowed):
        scanner.scan_point_start(122, extra={}) # start point 2x in a row not allowed
    scanner.scan_point_end()
    scanner.scan_point_start(123, extra={})
    scanner.scan_point_end()
    scanner.scan_end()
    with pytest.raises(AssertionError):
        scanner.roi_set([(100,150),(5,550)]) # 2nd bin starts below first bin
    scanner.roi_set([(100,150),(500,550), (600,650)])
    scanner.roi_start_counts()
    scanner.roi_get_counts()
    scanner.calibration_data_start(sample_id = 0, sample_name = "test_sample", routine = "ssrl_10_1_mix_cal")
    scanner.calibration_data_end()
    scanner.scan_start_calc_last_outputs(drift_correct_strategy="before_after_interp")
    scanner.file_end()
    with pytest.raises(statemachine.exceptions.TransitionNotAllowed):
        scanner.file_end() # end file while file ended not allowed


def test_scan():
    scan = Scan(var_name="mono", var_unit="eV", scan_num=0, beamtime_id="test_Beamtime", 
                ext_id=0, sample_id=0, sample_desc="test_desc", extra=None)
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
    with open(filename, "rb") as f:
        scan2 = Scan.from_json(f.read())
    assert scan == scan2
    scan.experiment_state_file_as_str(header=True)