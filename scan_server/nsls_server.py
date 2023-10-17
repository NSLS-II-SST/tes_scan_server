from scan_server import (TESScanner, DastardClient, DastardListener,
                         rpc_server, NSLSExtra, CringeDastardSettings)
import os
from pathlib import Path


# needed for exception filtering
import scan_server
import statemachine


def start():
    beamtime_id = 1
    base_user_output_dir = os.path.expanduser("~/userout")
    server_log_dir = os.path.expanduser("~/.scan_server")
    Path(server_log_dir).mkdir(parents=True, exist_ok=True)
    dastard_host = "localhost"
    dastard_port = 5500
    address = ""
    port = 4000
    time_human = rpc_server.time_human()
    cdsettings = CringeDastardSettings(
        record_nsamples=2000,
        record_npresamples=1000,
        trigger_threshold=-100,
        trigger_n_monotonic=6,
        write_ljh=True,
        write_off=True
    )

    no_traceback_error_types = [scan_server.dastard_client.DastardError,
                                statemachine.exceptions.TransitionNotAllowed]

    dastard_listener = DastardListener(dastard_host, dastard_port)
    dastard = DastardClient((dastard_host, dastard_port),
                            listener=dastard_listener)  # ,
    # pulse_trigger_params = None, noise_trigger_params = None)
    bg_log_file = open(os.path.join(server_log_dir, f"{time_human}_bg.log"), 'a')
    scanner = TESScanner(dastard, beamtime_id, base_user_output_dir,
                         bg_log_file, cdsettings=cdsettings)
    server_log_filename = os.path.join(server_log_dir, f"{time_human}.log")
    dispatch = rpc_server.get_dispatch_from(scanner)
    dispatch.update(rpc_server.get_dispatch_from(NSLSExtra()))
    print("Starting NSLS-II Scan Server")
    with open(server_log_filename, "w") as f:
        rpc_server.start(address, port, dispatch, verbose=True, log_file=f,
                         no_traceback_error_types=no_traceback_error_types)
