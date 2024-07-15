from .tes_base import TESModel, CringeDastardSettings
from .fake_dastard_client import FakeDastardClient
import os
from pathlib import Path

import rpc_server
from rpc_server import RPCDispatch, get_dispatch_from


def create_tes():
    beamtime_id = 1
    base_user_output_dir = "/tmp"
    server_log_dir = os.path.expanduser("~/.scan_server")
    Path(server_log_dir).mkdir(parents=True, exist_ok=True)
    time_human = rpc_server.time_human()
    cdsettings = CringeDastardSettings(
        record_nsamples=2000,
        record_npresamples=1000,
        trigger_threshold=-100,
        trigger_n_monotonic=6,
        write_ljh=True,
        write_off=True
    )

    # dastard_listener = DastardListener(dastard_host, dastard_port)
    dastard = FakeDastardClient(verbose=True)  # ,
    # pulse_trigger_params = None, noise_trigger_params = None)
    bg_log_file = open(os.path.join(server_log_dir, f"{time_human}_bg.log"), 'a')
    tes = TESModel(dastard, beamtime_id, base_user_output_dir,
                   bg_log_file, cdsettings)
    return tes


if __name__ == "__main__":
    rpc_host = "localhost"
    rpc_port = 4000
    tes = create_tes()
    rpc = RPCDispatch(rpc_host, rpc_port, get_dispatch_from(tes))
    rpc.start()
