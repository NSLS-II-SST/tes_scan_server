
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QApplication, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
import subprocess
import socket
import json
from .cringe_model import CringePowerOn, CringeAutotune
from .nsls_server import create_tes
from .rpc_server import RPCServer, get_dispatch_from


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
    def __init__(self, tes):
        super().__init__()
        self.tes = tes
        self.rpc = RPCServer("localhost", 4000, get_dispatch_from(tes))
        self.statusLabel = QLabel("Unknown")
        self.progButton = QPushButton("Start TES Programs")
        self.powerButton = QPushButton("Power TES On")
        self.dataButton = QPushButton("Start TES Data")
        self.tuneButton = QPushButton("Autotune TES")

        self.progButton.clicked.connect(lambda: self.startPrograms(self.progButton))
        self.powerButton.clicked.connect(lambda: self.tesPowerStart(self.powerButton))
        self.dataButton.clicked.connect(lambda: self.startData(self.dataButton))
        self.tuneButton.clicked.connect(lambda: self.startAutotune(self.tuneButton))

        #self.power_supplies = tower_power_supplies.TowerPowerSupplies()
        #self.scanner = ScannerComm("localhost", 4000)
        self.powerWorker = CringePowerOn()
        self.thread1 = QThread()
        self.powerWorker.moveToThread(self.thread1)
        self.powerWorker.finished.connect(self.tesPowerFinished)
        self.thread1.started.connect(self.powerWorker.run)

        self.thread2 = QThread()
        self.tuneWorker = CringeAutotune()
        self.tuneWorker.moveToThread(self.thread2)
        self.tuneWorker.finished.connect(self.autotuneFinished)
        self.thread2.started.connect(self.tuneWorker.run)

        self.thread3 = QThread()
        self.rpc.moveToThread(self.thread3)
        self.thread3.started.connect(self.rpc.start)
        self.thread3.start()

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.statusLabel)
        self.layout.addWidget(self.progButton)
        self.layout.addWidget(self.powerButton)
        self.layout.addWidget(self.dataButton)
        self.layout.addWidget(self.tuneButton)

        self.main = QWidget()
        self.main.setLayout(self.layout)
        self.setCentralWidget(self.main)

    @pyqtSlot(object, str)
    def printMsg(self, socket, msg):
        print(msg)

    def disableButtons(self):
        buttons = [self.progButton, self.powerButton, self.dataButton, self.tuneButton]
        for button in buttons:
            button.setEnabled(False)

    def enableButtons(self):
        buttons = [self.progButton, self.powerButton, self.dataButton, self.tuneButton]
        for button in buttons:
            button.setEnabled(True)

    def checkPrograms(self):
        programs = ["cringe", "dastard", "nsls_server", "dcom"]
        proc_returns = [subprocess.run(["pgrep", prog], stdout=subprocess.PIPE)
                        for prog in programs]
        for r, prog in zip(proc_returns, programs):
            if r.returncode != 1:
                return False
        return True

    def startPrograms(self, button):
        print("Start programs")
        subprocess.Popen(['open_tes_programs.sh'])
        button.setStyleSheet("background-color : grey")
        self.statusLabel.setText("TES Programs Starting")

        dlg = QMessageBox(self)
        dlg.setWindowTitle("TES Programs Starting")
        dlg.setText("Please wait until the CRINGE program starts. In this window, press 'Yes' if this was successful, and 'No' if it was not")
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        resp = dlg.exec()

        if resp == QMessageBox.Yes:
            button.setStyleSheet("background-color : green")
            self.statusLabel.setText("TES Programs Started")
        else:
            button.setStyleSheet("background-color : red")
            self.statusLabel.setText("TES Programs ")

    def tesPowerStart(self, button):
        print("Power on TES")
        self.statusLabel.setText("Waiting for TES Power On")
        self.disableButtons()
        self.thread1.start()

    def tesPowerFinished(self, resp):
        if 'ok' in resp:
            self.powerButton.setStyleSheet("background-color : green")
            self.statusLabel.setText("TES Powered on")
        else:
            self.powerButton.setStyleSheet("background-color : red")
            self.statusLabel.setText("Power On Failed, check Cringe running and try again")
        self.enableButtons()

    def startData(self, button):
        print("Start tes Data")
        response = self.tes.start_lancero()
        print(response)
        button.setStyleSheet("background-color : green")
        self.statusLabel.setText("TES Data started streaming")

    """
    def startData(self, button):
        print("Start tes Data")
        dlg = QMessageBox(self)
        dlg.setWindowTitle("TES Data Streaming")
        dlg.setText("Please find the DCOM window and click Start Data. In this window, press 'Yes' if this was successful, and 'No' if it was not")
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        resp = dlg.exec()

        if resp == QMessageBox.Yes:
            button.setStyleSheet("background-color : green")
            self.statusLabel.setText("TES Data started streaming")
        else:
            button.setStyleSheet("background-color : red")
            self.statusLabel.setText("TES Data is not streaming yet")
    """
    def startAutotune(self, button):
        print("Start autotune")
        self.statusLabel.setText("Running Cringe Autotune")
        self.disableButtons()
        self.thread2.start()

    def autotuneFinished(self, resp):
        if 'ok' in resp:
            self.tuneButton.setStyleSheet("background-color : green")
            self.statusLabel.setText("TES Autotuned")
        else:
            self.tuneButton.setStyleSheet("background-color : red")
            self.statusLabel.setText("Autotune failed, check Cringe window")
        self.enableButtons()


def main():
    import sys

    app = QApplication([])
    tes = create_tes()

    mainWindow = AutoTES(tes)
    mainWindow.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
