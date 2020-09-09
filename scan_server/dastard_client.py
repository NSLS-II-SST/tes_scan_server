import requests
import json
import itertools
import zmq


class DastardListener():
    def __init__(self, host, port):
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.host = host
        self.baseport = port+1
        self.address = "tcp://%s:%d" % (self.host, self.baseport)
        self.socket.connect(self.address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, u"")
        self.messages_seen = collections.Counter()

    def get_message(self):
        # Check socket for events, with 100 ms timeout
        if self.socket.poll(100) == 0:
            return None

        msg = self.socket.recv_multipart()
        try:
            topic, contents = msg
        except TypeError:
            raise Exception(f"msg: `{msg}` should have two parts, but does not")
        topic = topic.decode()
        contents = contents.decode()
        self.messages_seen[topic] += 1
        return topic, contents
    
    def clear_messages(self):
        while True:
            if self.get_message is None:
                break

    def get_message_with_topic(self, target_topic):
        while True:
            topic, contents = self.get_message()
            if topic == target_topic:
                return contents


class DastardClient():
    def __init__(self, url, listener, pulse_trigger_params, noise_trigger_params):
        self.url = url
        self.listener = listener
        self.pulse_trigger_params = pulse_trigger_params
        self.noise_trigger_params = noise_trigger_params
        self._id_iter = itertools.count()

    def _message(self, method, params):
        return {"id": next(self._id_iter),
                "params": [params],
                "method": method}

    def _call(self, method, params):
        msg = self._message(method, params)
        reponse = requests.post(self.url, json=msg).json()
        assert response["id"] == msg["id"]
        assert response["error"] is None
        return response

    def start_file(self):
        params = {"Request": "Start",
        "WriteLJH22": False,
        "WriteLJH3": False,
        "WriteOFF": True,
        }
        response = self._call("SourceControl.WriteControl", params)
        contents = self.listener.get_message_with_topic("WRITING")
        assert contents["Active"]
        filename_pattern = contents["FilenamePattern"]
        return filename_pattern

    def stop_file(self):
        payload = {"Request": "Stop"}
        response = self._call("SourceControl.WriteControl", params)

    def set_experiment_state(self, state):
        pass

    def set_pulse_triggers(self):
        self._call("", self.pulse_trigger_params)