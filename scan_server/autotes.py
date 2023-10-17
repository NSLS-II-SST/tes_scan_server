import PyQt5
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QApplication, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import subprocess
from cringe.cringe_control import CringeControl

class CringeWorkerBase(QObject):
    finished = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cc = CringeControl()

class CringePowerOn(CringeWorkerBase):
    def run(self):
        resp = self.cc.setup_crate()
        self.finished.emit(resp)

class CringeAutotune(CringeWorkerBase):
    def run(self):
        resp = self.cc.full_tune()
        self.finished.emit(resp)
        
class AutoTES(QMainWindow):
    def __init__(self):
        super().__init__()

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
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.statusLabel)
        self.layout.addWidget(self.progButton)
        self.layout.addWidget(self.powerButton)
        self.layout.addWidget(self.dataButton)
        self.layout.addWidget(self.tuneButton)
        
        self.main = QWidget()
        self.main.setLayout(self.layout)
        self.setCentralWidget(self.main)

    def disableButtons(self):
        buttons = [self.progButton, self.powerButton, self.dataButton, self.tuneButton]
        for button in buttons:
            button.setEnabled(False)

    def enableButtons(self):
        buttons = [self.progButton, self.powerButton, self.dataButton, self.tuneButton]
        for button in buttons:
            button.setEnabled(True)

    def startPrograms(self, button):
        print("Start programs")
        subprocess.Popen(['/home/xf07id1/autotes/test_open_programs.sh'])
        button.setStyleSheet("background-color : green")
        self.statusLabel.setText("TES Programs Started")

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
    mainWindow = AutoTES()
    mainWindow.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
