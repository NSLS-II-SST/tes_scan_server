from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QApplication,
    QVBoxLayout,
    QLabel,
    QMessageBox,
    QHBoxLayout,
    QCheckBox,
)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
import socket
import json
from functools import partial
from .nsls_server import create_tes
from .rpc_server import RPCDispatch, get_dispatch_from
from .epics_server import EpicsServer
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

class ScannerComm:
    def __init__(self, address, port):
        self.address = address
        self.port = port

    def formatMsg(self, method, *params, **kwargs):
        msg = {"method": method}
        if params is not None and params != []:
            msg["params"] = params
        if kwargs is not None and kwargs != {}:
            msg["kwargs"] = kwargs
        return json.dumps(msg).encode()

    def __getattr__(self, attr):
        def _method(*params, **kwargs):
            return self.sendrcv(attr, *params, **kwargs)

        return _method

    def sendrcv(self, method, *params, **kwargs):
        msg = self.formatMsg(method, *params, **kwargs)
        s = socket.socket()
        s.connect((self.address, self.port))
        s.send(msg)
        m = json.loads(s.recv(1024).decode())
        s.close()
        return m


class AutoTES(QMainWindow):
    def __init__(self, config_file):
        super().__init__()

        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        epics_config = config.pop("epics", {})

        tes = create_tes(config)
        
        self.tes = tes
        self.tesThread = QThread()
        self.tes.moveToThread(self.tesThread)
        self.tesThread.start()

        if self.tes._adrListener is None:
            self.hasADR = False
        else:
            self.hasADR = True

        if self.tes._cc is None:
            self.hasCringe = False
        else:
            self.hasCringe = True

        self.rpc = RPCDispatch("", 4000, get_dispatch_from(tes))
        self.statusLabel = QLabel("Unknown")

        self.setupButton = QPushButton("Setup all TES")
        self.progButton = QPushButton("Start TES Programs")

        self.dataButton = QPushButton("Start TES Data")

        if self.hasCringe:
            self.powerButton = QPushButton("Power TES On")
            self.tuneButton = QPushButton("Autotune TES")
            self.powerButton.clicked.connect(self.tesPowerStart)
            self.tuneButton.clicked.connect(self.startAutotune)
            self.tes.crate_powered_on.connect(self.tesPowerFinished)
            self.tes.autotuned.connect(self.autotuneFinished)
        
        if self.hasADR:
            self.autosetup = QCheckBox("Autosetup After Cycle")
            self.cryoStatus = QLabel("Unknown")
            self.autosetup.stateChanged.connect(self.sendAutosetup)
            self.tes._adrListener.stateChanged.connect(self.cryoStatus.setText)
            self.tes.autosetup_changed.connect(self.autosetup.setChecked)

        self.tesStatus = QLabel("Unknown")
        self.writingStatus = QLabel("Unknown")
        self.tesState = QLabel("Unknown")

        self.setupButton.clicked.connect(self.setupTES)
        self.progButton.clicked.connect(self.startPrograms)
        self.dataButton.clicked.connect(self.startData)
        self.tes.state_changed.connect(self.tesState.setText)
        self.tes.programs_started.connect(self.programsStarted)
        self.tes.source_on.connect(self.dataStarted)

        self.thread = QThread()
        self.rpc.moveToThread(self.thread)
        self.thread.started.connect(self.rpc.start)
        self.thread.start()

        self.epics_thread = QThread()
        self.epics_server = EpicsServer(self.tes, **epics_config)
        self.epics_server.moveToThread(self.epics_thread)
        self.epics_thread.started.connect(self.epics_server.start)
        self.epics_thread.start()


        mainLayout = QHBoxLayout()
        setupLayout = QVBoxLayout()
        setupLayout.addWidget(self.statusLabel)
        setupLayout.addWidget(self.setupButton)
        setupLayout.addWidget(self.progButton)
        if self.hasCringe:
            setupLayout.addWidget(self.powerButton)
        setupLayout.addWidget(self.dataButton)
        if self.hasCringe:
            setupLayout.addWidget(self.tuneButton)

        mainLayout.addLayout(setupLayout)

        statusLayout = QVBoxLayout()
        statusLayout.addWidget(QLabel("TES Status"))
        if self.hasADR:
            statusLayout.addWidget(self.autosetup)

        statusSubLayout = QHBoxLayout()
        statusLabels = QVBoxLayout()
        if self.hasADR:
            statusLabels.addWidget(QLabel("Cryostat:"))
        statusLabels.addWidget(QLabel("TES Ready:"))
        statusLabels.addWidget(QLabel("Writing:"))
        statusLabels.addWidget(QLabel("TES State:"))
        statusSubLayout.addLayout(statusLabels)

        statusReadout = QVBoxLayout()
        if self.hasADR:
            statusReadout.addWidget(self.cryoStatus)
        statusReadout.addWidget(self.tesStatus)
        statusReadout.addWidget(self.writingStatus)
        statusReadout.addWidget(self.tesState)

        statusSubLayout.addLayout(statusReadout)
        statusLayout.addLayout(statusSubLayout)
        mainLayout.addLayout(statusLayout)

        self.main = QWidget()
        self.main.setLayout(mainLayout)
        self.setCentralWidget(self.main)

    @pyqtSlot(object, str)
    def printMsg(self, socket, msg):
        print(msg)

    def disableButtons(self):
        buttons = [self.progButton, self.dataButton]
        if self.hasCringe:
            buttons.append(self.powerButton)
            buttons.append(self.tuneButton)

        for button in buttons:
            button.setEnabled(False)

    def enableButtons(self):
        buttons = [self.progButton, self.dataButton]
        if self.hasCringe:
            buttons.append(self.powerButton)
            buttons.append(self.tuneButton)
        for button in buttons:
            button.setEnabled(True)

    def checkPrograms(self):
        resp, err = self.rpc.call_method("check_programs_running")
        if err is None:
            return True
        else:
            return False

    @pyqtSlot(bool)
    def programsStarted(self, result):
        if result:
            self.progButton.setStyleSheet("background-color : green")
            self.statusLabel.setText("TES Programs Started")
        else:
            self.progButton.setStyleSheet("background-color : red")
            self.statusLabel.setText("Problem Starting TES Programs")

    def sendAutosetup(self, state):
        should_autosetup = self.autosetup.isChecked()
        self.rpc.call_method("autosetup", args=[should_autosetup])

    def setupTES(self):
        success, err = self.rpc.call_method("start_programs", kwargs={"restart": True})
        if not success:
            return
        if self.hasCringe:
            success, err = self.rpc.call_method("power_on_tes")
            if not success:
                return
        success, err = self.rpc.call_method("start_source", kwargs={"restart": True})
        if not success:
            return
        if self.hasCringe:
            success, err = self.rpc.call_method("autotune")

    def startPrograms(self):
        print("Start programs")
        self.progButton.setStyleSheet("background-color : grey")
        self.statusLabel.setText("TES Programs Starting")
        result, err = self.rpc.call_method("start_programs")
        if err is not None:
            print(err)

    def tesPowerStart(self, button):
        print("Power on TES")
        self.statusLabel.setText("Waiting for TES Power On")
        self.disableButtons()
        self.rpc.call_method("power_on_tes")

    @pyqtSlot(str)
    def tesPowerFinished(self, resp):
        if "ok" in resp:
            self.powerButton.setStyleSheet("background-color : green")
            self.statusLabel.setText("TES Powered on")
        else:
            self.powerButton.setStyleSheet("background-color : red")
            self.statusLabel.setText(
                "Power On Failed, check Cringe running and try again"
            )
        self.enableButtons()

    def startData(self):
        print("Start tes Data")
        response, err = self.rpc.call_method("start_source")
        print(response)

    @pyqtSlot(bool)
    def dataStarted(self, resp):
        if resp:
            self.dataButton.setStyleSheet("background-color : green")
            self.statusLabel.setText("TES Data started streaming")
        else:
            self.dataButton.setStyleSheet("background-color : red")
            self.statusLabel.setText("TES Data start failed")

    def startAutotune(self):
        print("Start autotune")
        self.statusLabel.setText("Running Cringe Autotune")
        self.disableButtons()
        self.rpc.call_method("autotune")

    def autotuneFinished(self, resp):
        if "ok" in resp:
            self.tuneButton.setStyleSheet("background-color : green")
            self.statusLabel.setText("TES Autotuned")
        else:
            self.tuneButton.setStyleSheet("background-color : red")
            self.statusLabel.setText("Autotune failed, check Cringe window")
        self.enableButtons()


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Start the AutoTES application with the given configuration file."
    )
    parser.add_argument("config_file", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    app = QApplication([])

    mainWindow = AutoTES(args.config_file)
    mainWindow.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
