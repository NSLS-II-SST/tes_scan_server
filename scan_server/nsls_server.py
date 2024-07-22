import argparse
from .dastard_client import DastardClient, DastardListener
from .tes_model import TESModel
from .rpc_server import RPCDispatch, get_dispatch_from

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def create_tes(config_file):

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    dastard_config = config.get("dastard")
    tes_config = config.get("tes")

    dastard_host = dastard_config.get("host")
    dastard_port = dastard_config.get("port")

    dastard_listener = DastardListener(dastard_host, dastard_port)
    dastard = DastardClient(
        (dastard_host, dastard_port), listener=dastard_listener, config=dastard_config
    )

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
    tes = create_tes(args.config_file)
    rpc = RPCDispatch(rpc_host, rpc_port, get_dispatch_from(tes))
    rpc.start()


if __name__ == "__main__":
    start()
