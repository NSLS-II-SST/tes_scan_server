import json
import itertools
import zmq
import socket
import collections
from collections import OrderedDict
import h5py
from typing import Union
import time
import numpy as np


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
        self.cache = {}

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
    
    def _update_messages(self):
        """get all messages form dastard, store them in self.cache"""
        while True:
            r = self.get_message()
            if r is None:
                # no message
                return None
            topic, contents = r
            self.cache[topic]=contents

    def get_message_with_topic(self, target_topic: str) -> Union[list, dict]:
        """ first update messages, which updates self.cache
        then retrieve the latest message for a given topic 
        from self.cache
        """
        # print("get_message_with_topic")
        self._update_messages()
        contents = self.cache[target_topic]
        return contents



class DastardError(Exception):
    pass

class DastardClient():
    """
    Assumptions: 
    1. Dastard has just been started.
    2. A source has just been stareted.
    3. Projectors have been loaded.
    4. Triggers have been set.
    5. (implied by above) Dastard is not writing.
    
    Potential Future:
    
    We could instead reduce the assumptions to:
    1. Dastard has a source running.
    2. Dastard is not writing.
    3. Projectors have been loaded. 
    4. Triggers have been set.

    We may also want TESScanner to load projectors and/or set triggers.
    """

    def __init__(self, addr_port, listener):
        self.addr_port = addr_port
        self.listener = listener
        self._id_iter = itertools.count()
        self._connect()
        self._request_status() # request one set of all messages on startup

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

    def _request_status(self):
        time.sleep(0.5) # make sure our zmq side it hooked up?
        self._call("SourceControl.SendAllStatus", "dummy")
        time.sleep(0.5) # wait to get all the statuses back

    def start_file(self, ljh22, off, path=None):
        params = {"Request": "Start",
        "WriteLJH22": ljh22,
        "WriteLJH3": False,
        "WriteOFF": off,
        }
        if path is not None:
            params["Path"] = path
        response = self._call("SourceControl.WriteControl", params)
        contents = self.listener.get_message_with_topic("WRITING")
        if not contents["Active"]:
            raise DastardError(f"Response from Dastard RPC should have contents[\"Active\"]=True, but it does not\ncontents:\n{contents}")
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
        "WaitForError": True}
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
        contents = self.listener.get_message_with_topic("WRITING") 
 

    def configure_record_lengths(self, npre, nsamp):
        params = {"Nsamp": nsamp,
        "Npre": npre}
        response = self._call("SourceControl.ConfigurePulseLengths", params)

    def get_data_path(self):
        return self.off_filename

    def set_projectors(self, projector_filename):
        source_type = self.get_source_type()
        if source_type.lower() == "lancero":
            channels_per_pixel = 2
        else:
            channels_per_pixel = 1
        print(f"set_projectors founrce source_type={source_type} and therefore channels_per_pixel={channels_per_pixel}")
        configs = getProjectorConfigs(projector_filename, self.get_name_to_number_index())
        success_chans = []
        failures = OrderedDict()
        for channelIndex, config in list(configs.items()):
            # print("sending ProjectorsBasis for {}".format(channelIndex))
            try:
                response = self._call(
                    "SourceControl.ConfigureProjectorsBasis", config)
                success_chans.append(channelIndex)
            except DastardError as ex:
                failures[channelIndex] = repr(ex)
                
        success = len(failures) == 0
        result = "success on channelIndices (not channelName): {}\n".format(
        sorted(success_chans)) + "failures:\n" + json.dumps(failures, sort_keys=True, indent=4)
        print("set_projectors result")
        print(result)

    def get_source_type(self):
        d = self.listener.get_message_with_topic("STATUS")
        return d["SourceName"]

    def get_n_channels(self):
        return len(self.get_name_to_number_index())

    def get_channel_names(self):
        # print("get_channel_names")
        d = self.listener.get_message_with_topic("CHANNELNAMES")
        # print(f"d={d}")
        channel_names = []  
        for name in d:
            channel_names.append(name)
        return channel_names

    # dastard channelNames go from chan1 to chanN and err1 to errN
    # we need to map from channelName to channelIndex (0-2N-1)
    def get_name_to_number_index(self):
        nameNumberToIndex = {}
        channel_names = self.get_channel_names()
        for (i, name) in enumerate(channel_names):
            if not name.startswith("chan"):
                continue
            nameNumber = int(name[4:])
            nameNumberToIndex[nameNumber] = i
            # for now since we only use this with lancero sources, error for non-odd index
            # if i % 2 != 1:
            #     raise Exception(
            #         "all fb channelIndices on a lancero source are odd, we shouldn't load projectors for even channelIndices")
        return nameNumberToIndex

    def set_pulse_trigger_all_chans(self, threshold, n_monotone):
        name_number_index = self.get_name_to_number_index()

        config = {'ChannelIndices': list(name_number_index.values()), 
        # 'AutoTrigger': False, 
        # 'AutoDelay': 0, 
        # 'LevelTrigger': False, 
        # 'LevelRising': False, 
        # 'LevelLevel': 0, 
        # 'EdgeTrigger': False, 
        # 'EdgeRising': False, 
        # 'EdgeFalling': False, 
        # 'EdgeLevel': 0, 
        'EdgeMulti': True, 
        # 'EdgeMultiNoise': False, 
        # 'EdgeMultiMakeShortRecords': False, 
        # 'EdgeMultiMakeContaminatedRecords': False, 
        # 'EdgeMultiDisableZeroThreshold': False, 
        'EdgeMultiLevel': -100, 
        'EdgeMultiVerifyNMonotone': 6}
        self._call("SourceControl.ConfigureTriggers", config)

    def set_noise_trigger_all_chans(self):
        name_number_index = self.get_name_to_number_index()

        config = {'ChannelIndices': list(name_number_index.values()), 
        'AutoTrigger': True, 
        'AutoDelay': 0, 
        # 'LevelTrigger': False, 
        # 'LevelRising': False, 
        # 'LevelLevel': 0, 
        # 'EdgeTrigger': False, 
        # 'EdgeRising': False, 
        # 'EdgeFalling': False, 
        # 'EdgeLevel': 0, 
        # 'EdgeMulti': True, 
        # 'EdgeMultiNoise': False, 
        # 'EdgeMultiMakeShortRecords': False, 
        # 'EdgeMultiMakeContaminatedRecords': False, 
        # 'EdgeMultiDisableZeroThreshold': False, 
        # 'EdgeMultiLevel': 100, 
        # 'EdgeMultiVerifyNMonotone': 1
        }
        self._call("SourceControl.ConfigureTriggers", config)

    def zero_all_triggers(self):
        # print("zero_all_triggers")
        channel_indicies_all = list(range(len(self.get_channel_names())))
        # print(f"channel_indicies_all={channel_indicies_all}")
        config = {
            "ChannelIndices": channel_indicies_all,
        }
        # print(f"config={config}")
        self._call("SourceControl.ConfigureTriggers", config)




def getProjectorConfigs(filename, nameNumberToIndex):
    """
    returns an OrderedDict mapping channel number to a dict for use in calling
    self.client.call("SourceControl.ConfigureProjectorsBasis", config)
    to set Projectors and Bases
    extracts the channel numbers and projectors and basis from the h5 file
    filename - points to a _model.hdf5 file created by Pope
    """
    out = OrderedDict()
    if not h5py.is_hdf5(filename):
        print(f"{filename} is not a valid hdf5 file")
        return out
    h5 = h5py.File(filename, "r")
    for key in list(h5.keys()):
        nameNumber = int(key)
        channelIndex = nameNumberToIndex[nameNumber]
        projectors = h5[key]["svdbasis"]["projectors"][()]
        basis = h5[key]["svdbasis"]["basis"][()]
        rows, cols = projectors.shape
        # projectors has size (n,z) where it is (rows,cols)
        # basis has size (z,n)
        # coefs has size (n,1)
        # coefs (n,1) = projectors (n,z) * data (z,1)
        # modelData (z,1) = basis (z,n) * coefs (n,1)
        # n = number of basis (eg 3)
        # z = record length (eg 4)
        nBasis = rows
        recordLength = cols
        if nBasis > recordLength:
            print("projectors transposed for dastard, fix projector maker")
            config = {
                "ChannelIndex": channelIndex,
                "ProjectorsBase64": toMatBase64(projectors.T)[0],
                "BasisBase64": toMatBase64(basis.T)[0],
            }
        else:
            config = {
                "ChannelIndex": channelIndex,
                "ProjectorsBase64": toMatBase64(projectors)[0],
                "BasisBase64": toMatBase64(basis)[0],
            }
        out[nameNumber] = config
    return out

def toMatBase64(array):
    """
    returns s,v
    s - a base64 encoded string containing the bytes in a format compatible with
    gonum.mat.Dense.MarshalBinary, header version 1
    v - the value that was base64 encoded, is of a custom np.dtype specific to the length of the projectors
    array - an np.array with dtype float64 (or convertable to float64)
    """
    nrow, ncol = array.shape
    dt = np.dtype([('version', np.uint32), ('magic', np.uint8, (4,)), ("nrow", np.int64),
                   ("ncol", np.int64), ("zeros", np.int64, 2), ("data", np.float64, nrow*ncol)])
    a = np.array([(1, [ord("G"), ord("F"), ord("A"), 0], nrow, ncol, [0, 0], array.ravel())], dt)
    s_bytes = base64.b64encode(a)
    s = s_bytes.decode(encoding="ascii")
    return s, a[0]