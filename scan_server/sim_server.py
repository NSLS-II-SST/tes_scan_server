from .tes_base import TESModel, CringeDastardSettings
from .fake_dastard_client import FakeDastardClient
from .rpc_server import RPCDispatch, get_dispatch_from, time_human
import os
from pathlib import Path



def create_tes():
    beamtime_id = 1
    base_user_output_dir = "/tmp"
    server_log_dir = os.path.expanduser("~/.scan_server")
    Path(server_log_dir).mkdir(parents=True, exist_ok=True)
    log_time = time_human()
    cdsettings = CringeDastardSettings(
        record_nsamples=2000,
        record_npresamples=1000,
        trigger_threshold=-100,
        trigger_n_monotonic=6,
        write_ljh=True,
        write_off=True,
        projector_filename=""
    )

    # dastard_listener = DastardListener(dastard_host, dastard_port)
    dastard = FakeDastardClient(verbose=True)  # ,
    # pulse_trigger_params = None, noise_trigger_params = None)
    bg_log_file = open(os.path.join(server_log_dir, f"{log_time}_bg.log"), 'a')
    tes = TESModel(dastard, beamtime_id, base_user_output_dir,
                   bg_log_file, cdsettings)
    return tes


def start():
    #app = QApplication([])
    rpc_host = "localhost"
    rpc_port = 4000
    tes = create_tes()
    rpc = RPCDispatch(rpc_host, rpc_port, get_dispatch_from(tes))
    rpc.start()


if __name__ == "__main__":
    start()
