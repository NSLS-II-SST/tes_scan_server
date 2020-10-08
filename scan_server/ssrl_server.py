from scan_server import TESScanner, Scan, DastardClient, DastardListener, CalibrationLog, rpc_server
import os
from pathlib import Path

# needed for exception filtering
import scan_server
import statemachine



def start():
    beamtime_id = 1
    base_user_output_dir = "/var/folders/_0/25kp6h7x25v6vyjv2yjlcnkm000wrm/T"
    server_log_dir = os.path.expanduser("~/.scan_server")
    Path(server_log_dir).mkdir(parents=True, exist_ok=True)
    dastard_host = "localhost"
    dastard_port = 5500
    address = "localhost"
    port = 4000
    time_human = rpc_server.time_human()

    no_traceback_error_types = [scan_server.dastard_client.DastardError, statemachine.exceptions.TransitionNotAllowed]

    dastard_listener = DastardListener(dastard_host, dastard_port)
    dastard = DastardClient((dastard_host, dastard_port), listener = dastard_listener,
        pulse_trigger_params = None, noise_trigger_params = None)
    bg_log_file = os.path.join(server_log_dir, f"{time_human}_bg.log")
    scanner = TESScanner(dastard, beamtime_id, base_user_output_dir, bg_log_file)
    server_log_filename = os.path.join(server_log_dir, f"{time_human}.log")
    dispatch = rpc_server.get_dispatch_from(scanner)
    print("Starting SSRL Scan Server")
    with open(server_log_filename, "w") as f:
        rpc_server.start(address, port, dispatch, verbose=True, log_file=f, 
            no_traceback_error_types=no_traceback_error_types)


