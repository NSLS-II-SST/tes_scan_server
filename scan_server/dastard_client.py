import json
import itertools
import zmq
import socket
import collections

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
            topic, contents_str = msg
        except TypeError:
            raise Exception(f"msg: `{msg}` should have two parts, but does not")
        topic = topic.decode()
        contents = json.loads(contents_str.decode())
        self.messages_seen[topic] += 1
        return topic, contents
    
    def clear_messages(self):
        while True:
            if self.get_message is None:
                break

    def get_message_with_topic(self, target_topic: str) -> dict:
        while True:
            topic, contents = self.get_message()
            if topic == target_topic:
                if not isinstance(contents, dict):
                    raise DastardError(f"contents should be a dict, is a {type(contents)}. contents={contents}")
                return contents


class DastardError(Exception):
    pass

class DastardClient():
    def __init__(self, addr_port, listener):
        self.addr_port = addr_port
        self.listener = listener
        self._id_iter = itertools.count()
        self._connect()

    def _connect(self):
        try:
            self._socket = socket.create_connection(self.addr_port)
        except socket.error as ex:
            host, port = self.addr_port
            raise Exception(f"Could not connect to Dastard at {host}:{port}")

    def _message(self, method_name, params):
        if not isinstance(params, list):
            params = [params]
        d = {"id": next(self._id_iter),
        "params": params,
        "method": method_name} 
        return d

    def _call(self, method_name: str, params, verbose=True):
        msg = self._message(method_name, params)
        if verbose: 
            print(f"Dastard Client: sending: {msg}")
        else:
            print(f"Dastard Client: calling {method_name}")
        self._socket.sendall(json.dumps(msg).encode())
        response = self._socket.recv(4096)
        response = json.loads(response.decode())
        if verbose:     
            print(f"Dastard Client: response: {response}")
        else:
            print(f"Dastard Client: got response for {method_name}")
        if not response["id"] == msg["id"]:
            raise DastardError(f"response id does not match message id")
        err = response.get("error", None)
        if err is not None:
            raise DastardError(f"""Dastard responded with error: {err}""") 
        return response["result"]

    def start_file(self):
        params = {"Request": "Start",
        "WriteLJH22": True,
        "WriteLJH3": False,
        "WriteOFF": False,
        }
        response = self._call("SourceControl.WriteControl", params)
        contents = self.listener.get_message_with_topic("WRITING")
        if not contents["Active"]:
            raise DastardError(f"Response from Dastard RPC should have contents[\"Active\"]=True, but it does not\nconents:\n{contents}")
        self.off_filename = contents["FilenamePattern"]%("chan1","off")
        return self.off_filename


    def stop_source(self):
        response = self._call("SourceControl.Stop", "")

    def configure_simulate_pulse_source(self, nchan, sample_rate_hz, pedestal, amplitudes, samples_per_pulse):
        params = {"Nchan": nchan,
        "SampleRate" : sample_rate_hz,
        "Pedestal": pedestal,
        "Amplitudes": amplitudes,
        "Nsamp": samples_per_pulse}
        response = self._call("SourceControl.ConfigureSimPulseSource", params)

    def start_sim_pulse_source(self):
        response = self._call("SourceControl.Start", "SIMPULSESOURCE")

    def set_experiment_state(self, state):
        params = {"Label": state,
        "WaitForError": False}
        response = self._call("SourceControl.SetExperimentStateLabel", params)

    def set_triggers(self, full_trigger_state):
        response = self._call("SourceControl.ConfigureTriggers", full_trigger_state)

    def start_writing(self, ljh22, off, path=None):
        params = {"Request": "Start",
        "WriteLJH22": ljh22,
        "WriteOff": off}
        if path is not None:
            params["Path"] = path
        response = self._call("SourceControl.WriteControl", params)
    
    def stop_writing(self):
        params = {"Request": "Stop"}
        response = self._call("SourceControl.WriteControl", params)

    def configure_record_lengths(self, npre, nsamp):
        params = {"Nsamp": nsamp,
        "Npre": npre}
        response = self._call("SourceControl.ConfigurePulseLengths", params)

    def get_data_path(self):
        return self.off_filename

