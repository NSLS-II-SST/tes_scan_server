from .dastard_client import DastardClient, DastardListener
from .tes_model import TESModel, CringeDastardSettings
import os
from pathlib import Path
from .rpc_server import RPCDispatch, get_dispatch_from, time_human
from .cringe_model import CringeControl


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def create_tes(config_file):

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    dastard_config = config.get("dastard")
    tes_config = config.get("tes")

    beamtime_id = 1
    base_user_output_dir = os.path.expanduser("~/userout")
    server_log_dir = os.path.expanduser("~/.scan_server")
    Path(server_log_dir).mkdir(parents=True, exist_ok=True)
    dastard_host = dastard_config.get("host")
    dastard_port = dastard_config.get("port")

    time_human_readable = time_human()
    cdsettings = CringeDastardSettings(
        record_nsamples=2000,
        record_npresamples=1000,
        trigger_threshold=-100,
        trigger_n_monotonic=6,
        write_ljh=True,
        write_off=True,
        projector_filename=os.path.expanduser("~/.scan_server/nsls_projectors.hdf5"),
    )

    dastard_listener = DastardListener(dastard_host, dastard_port)
    dastard = DastardClient(
        (dastard_host, dastard_port), listener=dastard_listener, config=dastard_config
    )
    bg_log_file = open(
        os.path.join(server_log_dir, f"{time_human_readable}_bg.log"), "a"
    )
    cc = CringeControl()
    tes = TESModel(
        dastard, tes_config
    )  # , beamtime_id, base_user_output_dir, bg_log_file, cdsettings)
    return tes


def start():
    rpc_host = ""
    rpc_port = 4000
    tes = create_tes()
    rpc = RPCDispatch(rpc_host, rpc_port, get_dispatch_from(tes))
    rpc.start()


if __name__ == "__main__":
    start()
