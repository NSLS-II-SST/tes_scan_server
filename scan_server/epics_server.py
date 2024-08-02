from pcaspy import SimpleServer, Driver
from pcaspy.driver import manager
from PyQt5.QtCore import QObject
from functools import partial


pvdb = {
    "STATE": {"type": "str", "value": ""},
    "FILENAME": {"type": "char", "value": "", "count": 255},
    "SCAN_STR": {"type": "char", "value": "", "count": 255},
    "SCAN_NUM": {"type": "int", "value": 0},
    "NOISE_UID": {"type": "char", "value": "", "count": 255},
    "PROJECTOR_UID": {"type": "char", "value": "", "count": 255},
    "RSYNC_ON_FILE_END": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "RSYNC_ON_SCAN_END": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "WRITE_LJH": {"type": "enum", "enums": ["False", "True"], "value": 0},
    "WRITE_OFF": {"type": "enum", "enums": ["False", "True"], "value": 0},
}


class TESDriver(Driver):
    def __init__(self):
        super().__init__()

    def write(self, reason, value):
        super().write(reason, value)
        self.updatePV(reason)


class EpicsServer(QObject):
    def __init__(self, tes, prefix="SIM_TES:", parent=None, **kwargs):
        super().__init__(parent=parent)
        self.server = SimpleServer()
        self.server.createPV(prefix, pvdb)
        self.drv = TESDriver()
        self.tes = tes

        # Connect signals and initialize PV values
        self.connect_and_initialize("state", "STATE")
        self.connect_and_initialize("filename", "FILENAME")
        self.connect_and_initialize("scan_str", "SCAN_STR")
        self.connect_and_initialize("scan_num", "SCAN_NUM")
        self.connect_and_initialize("noise_uid", "NOISE_UID")
        self.connect_and_initialize("projector_uid", "PROJECTOR_UID")
        self.connect_and_initialize("rsync_on_file_end", "RSYNC_ON_FILE_END")
        self.connect_and_initialize("rsync_on_scan_end", "RSYNC_ON_SCAN_END")
        self.connect_and_initialize("write_ljh", "WRITE_LJH")
        self.connect_and_initialize("write_off", "WRITE_OFF")
        self.connect_and_initialize("dastard_connected", "CONNECTED")

        print("EpicsServer started")

    def connect_and_initialize(self, tes_attr, pv_name):
        # Connect the signal
        signal = getattr(self.tes, f"{tes_attr}_changed")
        signal.connect(partial(self.drv.write, pv_name))

        # Initialize the PV value
        initial_value = getattr(self.tes, tes_attr)
        self.drv.write(pv_name, initial_value)

    def start(self):
        print("TES EpicsServer serving PVs:")
        for pv in manager.pvf:
            print(pv)
        while True:
            self.server.process(0.5)
