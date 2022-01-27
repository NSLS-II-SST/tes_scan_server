import instruments
from cringe.cringe_control import CringeControl

import socket


# class Agilent33500SCPI():
#     def __init__(self, addr="192.168.101.59", port=5025):
#         self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.s.settimeout(1)
#         self.s.connect((addr, port))
    
#     def write(self, msg):
#         self.s.send(f"{msg}\r\n".encode())

#     def read(self):
#         return self.s.read(1024).decode()
        
#     def ask(self, msg):
#         self.write(msg)
#         return self.read()

class NSLSExtra():
    def __init__(self):
        # self._a = Agilent33500SCPI()
        self._cc = CringeControl()
        self._towerps1 = instruments.AgilentE3631A("towerps1")

    # def set_awg_out2(self, v):
    #     self._a.write(f"SOUR2:VOLT:OFFS {v:f}")

    def set_detector_bias_all(self, dac):
        self._cc.set_tower_card_all_channels("DB1", int(dac))

    def set_magnetic_field(self, v):
        # polarity controlled by bananna plug polarity
        self._towerps1.setCurrentLimit(output="P6V", voltage=v, amps_limit=0.1)

