from scan_server import TESScanner, rpc_server
from scan_server.fake_dastard_client import FakeDastardClient
import os
from pathlib import Path

# needed for exception filtering
import scan_server
import statemachine

def start():
    beamtime_id = 1
    base_user_output_dir = "/tmp"
    server_log_dir = os.path.expanduser("~/.scan_server")
    Path(server_log_dir).mkdir(parents=True, exist_ok=True)
    dastard_host = "localhost"
    dastard_port = 5500
    address = ""
    port = 4000
    time_human = rpc_server.time_human()

    no_traceback_error_types = [scan_server.dastard_client.DastardError, statemachine.exceptions.TransitionNotAllowed]

    #dastard_listener = DastardListener(dastard_host, dastard_port)
    dastard = FakeDastardClient(verbose=True)#,
    #pulse_trigger_params = None, noise_trigger_params = None)
    bg_log_file = open(os.path.join(server_log_dir, f"{time_human}_bg.log"), 'a')
    scanner = TESScanner(dastard, beamtime_id, base_user_output_dir, bg_log_file)
    server_log_filename = os.path.join(server_log_dir, f"{time_human}.log")
    dispatch = rpc_server.get_dispatch_from(scanner)
    print("Starting Simulated Scan Server")
    with open(server_log_filename, "w") as f:
        rpc_server.start(address, port, dispatch, verbose=True, log_file=f, 
            no_traceback_error_types=no_traceback_error_types)

