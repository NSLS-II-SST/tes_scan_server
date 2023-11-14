import zmq
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

class ADRListener(QObject):
    newCycle = pyqtSignal(str)
    stateChanged = pyqtSignal(str)
    def __init__(self, host='localhost', port=5021):
        super().__init__()
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.host = host
        self.baseport = port
        self.address = "tcp://%s:%d" % (self.host, self.baseport)
        self.socket.connect(self.address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, u"")
        self.cache = {}
        self.timer = QTimer()

    def get_message(self):
        # Check socket for events, with 100 ms timeout
        if self.socket.poll(10) == 0:
            return None
        
        msg = self.socket.recv_json()
        for k, v in msg.items():
            if k == 'state':
                if self.cache.get(k, "") != v:
                    self.stateChanged.emit(v)
                    self.cache[k] = v
            if k == 'uid':
                if self.cache.get(k, "") != v:
                    self.newCycle.emit(v)
                    self.cache[k] = v
            else:
                self.cache[k] = v
        return
    
    def start(self):
        self.timer.timeout.connect(self.get_message)
        self.timer.start(500)

    def get_message_with_topic(k):
        return self.cache.get(k, None)
