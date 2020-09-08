from statemachine import StateMachine, State
import mass
from typing import List



class Scan():
    def __init__(self, var_names: List[str], scan_num: int, beamtime_id: str, ext_id: str):
        self.beamtime_id = beamtime_id
        self.ext_id = ext_id
        self.var_names = var_names
        self.scan_num = scan_num
        self.var_values = []
        self.experiment_state_labels = []

    def add_point(self, vars_dict):
        assert len(vars_dict) == len(self.var_names)
        values = np.array([vars_dict[name] for name in self.var_names])
        point_num = len(self.var_values)
        self.var_values.append(values)
        experiment_state_label = f"SCAN{point_num}"
        self.experiment_state_labels.append(experiment_state_label)
        return experiment_state_label    

    def end_point(self):
        self.to_disk()

    def to_disk(self):
        return


class ScannerState(StateMachine):
    """defines allowed state transitions"""
    no_file = State('no_file', initial=True)
    pause = State("pause")
    scan = State('scan')
    scan_point = State("scan_point")
    cal_data = State('cal_data')

    scan_start = pause.to(scan)
    scan_end = scan.to(pause)

    scan_point_start = scan.to(scan_point)
    scan_point_end = scan_point.to(scan)

    file_end = pause.to(no_file)
    file_start = no_file.to(pause)

    cal_data_start = pause.to(cal_data)
    cal_data_stop = cal_data.to(pause)


class TESScanner():
    """talks to dastard and mass"""
    def __init__(self, client, pulse_trigger_msg, noise_trigger_msg, beamtime_id: str):
        self.client = client
        self.pulse_trigger_msg = pulse_trigger_msg
        self.noise_trigger_msg = noise_trigger_msg
        self.beamtime_id = beamtime_id
        self.state: ScannerState = ScannerState()
        self.reset()

    def reset(self):
        self.last_scan = None
        self.scan = None
        self.data = None
        self.roi_counts_start = None
        self.rois_bin_edges = None
        self.calibration_to_routine: List[str] = [] 
        self.next_cal_number: int = 0

    def file_start(self):
        self.state.file_start()
        self.filename = self.client.start_file()
        self.client.send_experiment_state("PAUSE")
        # self.data = mass.ChannelGroup(self.filename)

    def calibration_data_start(self, sample_id: int, sample_name: str, routine: str):
        self.set_pulse_triggers()
        self.state.scan_start()
        self.set_experiment_state(f"CALIBRATION{self.next_cal_number}")
        self.calibration_to_routine.append(routine)
        self.next_cal_number += 1
        assert len(self.calibration_to_routine) == self.next_cal_number

    def calibration_data_end(self):
        self.state.scan_end()
        self.set_experiment_state("PAUSE")
        self.data.update_from_disk()

    def calibration_learn_from_last_data(self, cal_number: int):
        last_cal_number = self.next_cal_number - 1
        routine = self.calibration_to_routine[self.last_calname]
        result = self.apply_routine(routine, cal_number, self.data)
        result.error_if_energy_resolution_too_bad()
        # now we can access realtime_energy

    def roi_set(self, rois_list):
        # roi list is a a list of pairs of lo, hi energy pairs
        bin_edges = []
        for (lo_ev, hi_ev) in rois_list:
            assert hi_ev > lo_ev
            if len(bin_edges) > 0:
                assert lo_ev > bin_edges[-1]
            bin_edges.append(lo_ev)
            bin_edges.append(hi_ev)
        self.rois_bin_edges = np.array(bin_edges)
    
    def roi_start_counts(self):
        self.roi_counts_start = time.time()
    
    def roi_get_counts(self):
        "first call set_rois, then start_rois_counts, then this will return counts in all rois since last call to start_rois_counts"
        t = self.roi_counts_start
        bin_centers, counts = self.data.hist("realtime_energy", self.rois_bin_edges)
        return counts[::2]

    def scan_define(self, name, ext_id, scan_vars, sample_id, sample_desc, extra):
        self.state.scan_start()
        self.scan = Scan(name, ext_it, scan_vars, sample_id, sample_desc, extra)

    def scan_point_start(self, scan_vars_dict):
        self.state.scan_point_start()
        experiment_state_label = self.scan.add_point(scan_vars_dict)
        self.set_experiment_state(experiment_state_label)

    def scan_point_end(self):
        self.state.scan_point_end()
        self.scan.end_point()
        self.set_experiment_state("PAUSE")

    def scan_end(self):
        self.state.scan_end()
        self.scan.end()
        self.last_scan = self.scan
        self.scan = None
        
    def scan_start_calc_last_outputs(self, drift_correct_strategy):
        self.validate_drift_correct_strategy(drift_correct_strategy)
        self.launch_process_to_calc_outputs(drift_correct_strategy)

    def file_end(self):
        self.state.file_end()
        self.reset()
    





# class ScanServer():
#     """talks to beamline and TESScanner"""
#     def __init__(self, port, command_resolver):
#         self.dastard_client = None
#         self.socket = None

#     def get_command(self):
#         try:
#             return self.socket.read()
#         except TimeoutError:
#             return None

#     def run(self):
#         while True:
#             cmd = self.get_command()
#             if cmd is not None:
#                 try:
#                     response = self.resolve_command(cmd)
#                     self.log(cmd, response)
#                     self.respond(response)
#                 except Exception as ex:
#                     error_desc = get_backtrace()
#                     self.log(cmd, error_desc)
#                     sys.exit()
#             time.sleep(0.01)
    
#     def resolve_command(self, cmd):
#         command_resolver(cmd)