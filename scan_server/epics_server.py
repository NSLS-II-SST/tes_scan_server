from pcaspy import SimpleServer, Driver
from pcaspy.driver import manager
from PyQt5.QtCore import QObject, pyqtSlot
from functools import partial


pvdb = {
    "STATE": {"type": "str", "value": ""},
    "FILENAME": {"type": "char", "value": "", "count": 255},
    "SCAN_STR": {"type": "char", "value": "", "count": 255},
    "SCAN_NUM": {"type": "int", "value": 0},
    "NOISE_UID": {"type": "char", "value": "", "count": 255},
    "PROJECTOR_UID": {"type": "char", "value": "", "count": 255},
    "CALIBRATION_UID": {"type": "char", "value": "", "count": 255},
    "NOISE_FILE": {"type": "char", "value": "", "count": 255},
    "PROJECTOR_FILE": {"type": "char", "value": "", "count": 255},
    "RSYNC_ON_FILE_END": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "RSYNC_ON_SCAN_END": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "WRITE_LJH": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "WRITE_OFF": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "CONNECTED": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "PROJECTORS": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "RUNNING": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "STATUS": {
        "type": "enum",
        "enums": [
            "Not Running",
            "Needs Noise Data",
            "Needs Projector Data",
            "Needs Projectors Loaded",
            "Ready for Calibration",
            "Ready for Data",
        ],
        "value": 0,
    },
}


class TESDriver(Driver):
    def __init__(self):
        super().__init__()

    def write(self, reason, value):
        super().write(reason, value)
        self.updatePV(reason)
        if reason in [
            "RUNNING",
            "NOISE_UID",
            "PROJECTOR_UID",
            "PROJECTORS",
            "CALIBRATION_UID",
        ]:
            self.update_status()

    def update_status(self):
        running = self.getParam("RUNNING")
        noise_uid = self.getParam("NOISE_UID")
        projector_uid = self.getParam("PROJECTOR_UID")
        projector_loaded = self.getParam("PROJECTORS")
        calibration_uid = self.getParam("CALIBRATION_UID")

        if not running:
            status = 0  # "Not Running"
        elif noise_uid == "":
            status = 1  # "Needs Noise Data"
        elif projector_uid == "":
            status = 2  # "Needs Projector Data"
        elif not projector_loaded:
            status = 3
        elif calibration_uid == "":
            status = 4  # "Ready for Calibration"
        else:
            status = 5  # "Running"

        self.setParam("STATUS", status)
        self.updatePV("STATUS")


class EpicsServer(QObject):
    def __init__(self, tes, prefix="SIM_TES:", parent=None, **kwargs):
        super().__init__(parent=parent)
        self.server = SimpleServer()
        self.server.createPV(prefix, pvdb)
        self.drv = TESDriver()
        self.tes = tes

        # Connect signals and initialize PV values
        self.connect_and_initialize("filename", "FILENAME")
        self.connect_and_initialize("state", "STATE")
        self.connect_and_initialize("scan_str", "SCAN_STR")
        self.connect_and_initialize("scan_num", "SCAN_NUM")
        self.connect_and_initialize("noise_uid", "NOISE_UID")
        self.connect_and_initialize("noise_file", "NOISE_FILE")
        self.connect_and_initialize("projector_file", "PROJECTOR_FILE")
        self.connect_and_initialize("projector_uid", "PROJECTOR_UID")
        self.connect_and_initialize("calibration_uid", "CALIBRATION_UID")
        self.connect_and_initialize("rsync_on_file_end", "RSYNC_ON_FILE_END")
        self.connect_and_initialize("rsync_on_scan_end", "RSYNC_ON_SCAN_END")
        self.connect_and_initialize("write_ljh", "WRITE_LJH")
        self.connect_and_initialize("write_off", "WRITE_OFF")
        self.connect_and_initialize("dastard_connected", "CONNECTED")
        self.connect_and_initialize("running", "RUNNING")
        self.connect_and_initialize("projectors", "PROJECTORS")

        # Initialize status PV
        self.drv.update_status()

        print("EpicsServer started")

    def connect_and_initialize(self, tes_attr, pv_name):
        # Connect the signal
        # print(f"Attempting to connect and initialize {tes_attr} to PV {pv_name}")

        signal = getattr(self.tes, f"{tes_attr}_changed")
        # print(f"Signal for {tes_attr}: {signal}")

        def writePV(value):
            # print(f"WritePV called for {pv_name} with value: {value}")
            self.drv.write(pv_name, value)

        signal.connect(writePV)

        # Initialize the PV value
        initial_value = getattr(self.tes, tes_attr)
        self.drv.write(pv_name, initial_value)

    def start(self):
        print("TES EpicsServer serving PVs:")
        for pv in manager.pvf:
            print(pv)
        while True:
            self.server.process(0.5)
