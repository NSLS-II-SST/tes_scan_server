from qtpy.QtCore import QObject, Signal
try:
    from cringe.cringe_control import CringeControl
except ImportError:
    class CringeControl:
        def __init__(self):
            print("No Cringe, Running Simulation Mode for GUI Testing")

        def setup_crate(self):
            print("CRINGE SIM MODE, SETUP NOT RUN")
            return "ok"

        def full_tune(self):
            print("CRINGE SIM MODE, TUNE NOT RUN")
            return "ok"


class CringeWorkerBase(QObject):
    finished = Signal(str)

    def __init__(self):
        super().__init__()
        self.cc = CringeControl()


class CringePowerOn(CringeWorkerBase):
    def run(self):
        resp = self.cc.setup_crate()
        self.finished.emit(resp)
        return resp


class CringeAutotune(CringeWorkerBase):
    def run(self):
        resp = self.cc.full_tune()
        self.finished.emit(resp)
        return resp
