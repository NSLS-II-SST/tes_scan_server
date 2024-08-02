import argparse
from .dastard_client import DastardClient, DastardListener
from .tes_model import TESModel
from .rpc_server import RPCDispatch, get_dispatch_from
from .epics_server import EpicsServer
from PyQt5.QtCore import QThread, QCoreApplication
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def create_tes(config):

    dastard_config = config.get("dastard")
    tes_config = config.get("tes")

    dastard_host = dastard_config.get("host")
    dastard_port = dastard_config.get("port")

    # dastard_listener = DastardListener(dastard_host, dastard_port)
    dastard = DastardClient((dastard_host, dastard_port), config=dastard_config)

    if tes_config.get("cringe", False):
        from .cringe_model import CringeControl

        cringe = CringeControl()
    else:
        cringe = None

    if tes_config.get("adr", False):
        from .adr_model import ADRListener

        adr = ADRListener()
    else:
        adr = None

    tes = TESModel(
        dastard, tes_config, adr, cringe
    )  # , beamtime_id, base_user_output_dir, bg_log_file, cdsettings)
    return tes


def start():
    parser = argparse.ArgumentParser(
        description="Start the TES server with the given configuration file."
    )
    parser.add_argument("config_file", type=str, help="Path to the configuration file")
    parser.add_argument("--host", type=str, default="", help="RPC server host")
    parser.add_argument("--port", type=int, default=4000, help="RPC server port")
    args = parser.parse_args()

    rpc_host = args.host
    rpc_port = args.port

    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)

    epics_config = config.pop("epics", {})

    app = QCoreApplication(sys.argv)

    tes = create_tes(config)

    # Create QCoreApplication

    # Create and start RPC server thread
    rpc_thread = QThread()
    rpc = RPCDispatch(rpc_host, rpc_port, get_dispatch_from(tes))
    rpc.moveToThread(rpc_thread)
    rpc_thread.started.connect(rpc.start)
    rpc_thread.start()

    # Create and start EPICS server thread
    epics_thread = QThread()
    epics_server = EpicsServer(tes, **epics_config)
    epics_server.moveToThread(epics_thread)
    epics_thread.started.connect(epics_server.start)
    epics_thread.start()
    print("TES Server started, ctrl-\\ to stop")
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    start()
