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
import base64
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer


class DastardListener(QObject):
    message_received = pyqtSignal(str, object)

    def __init__(self, host, port):
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.host = host
        self.baseport = port + 1
        self.address = f"tcp://{self.host}:{self.baseport}"
        self.socket.connect(self.address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.messages_seen = collections.Counter()
        self.cache = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_messages)

    def start(self):
        self.timer.start(1000)  # Update every 1 second

    def stop(self):
        self.timer.stop()

    def update_messages(self):
        while True:
            try:
                topic, contents = self.get_message()

                if topic is None:
                    break
                self.cache[topic] = contents
                self.message_received.emit(topic, contents)
            except Exception as e:
                print(f"Error updating messages: {e}")

                break

    def get_message(self):
        if self.socket.poll(100) == 0:
            return None, None

        msg = self.socket.recv_multipart()
        try:
            topic, contents_str = msg
        except TypeError:
            raise Exception(f"msg: `{msg}` should have two parts, but does not")
        topic = topic.decode()
        contents = json.loads(contents_str.decode())
        self.messages_seen[topic] += 1
        return topic, contents

    def get_message_with_topic(self, target_topic: str):
        return self.cache.get(target_topic)

    def reset(self):
        print("Reset - draining messages")
        self.update_messages()
        print("Reset - messages drained")
        self.messages_seen = collections.Counter()
        self.cache = {}


class DastardError(Exception):
    pass


class SignalProperty:
    def __init__(self, default_value):
        self.value = default_value
        self.signal = None

    def __set_name__(self, obj, name):
        self.name = name
        self.signal_name = f"{name}_changed"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.value

    def __set__(self, obj, value):
        if self.value != value:
            self.value = value
            signal = getattr(obj, self.signal_name, None)

            if signal is not None:
                signal.emit(value)


class DastardClient(QObject):
    state_changed = pyqtSignal(str)
    writing_changed = pyqtSignal(bool)
    channel_names_changed = pyqtSignal(list)
    status_updated = pyqtSignal(dict)
    source_changed = pyqtSignal(str)
    running_changed = pyqtSignal(bool)
    filename_changed = pyqtSignal(str)
    connected_changed = pyqtSignal(bool)
    disabled_changed = pyqtSignal(list)
    projectors_changed = pyqtSignal(bool)

    # SignalProperty attributes
    writing = SignalProperty(False)
    channel_names = SignalProperty([])
    source = SignalProperty("None")
    running = SignalProperty(False)
    connected = SignalProperty(False)
    disabled_chans = SignalProperty([])
    projectors = SignalProperty(False)

    def __init__(self, addr_port, config={}):
        super().__init__()
        self.addr_port = addr_port
        self.config = config
        self._id_iter = itertools.count()
        self._connect()

        self.listener_thread = QThread()
        self.listener = DastardListener(*self.addr_port)
        self.listener.moveToThread(self.listener_thread)
        self.listener_thread.started.connect(self.listener.start)
        self.listener.message_received.connect(self.handle_message)
        self.listener_thread.start()

        self._off_filename = ""

        if self.connected:
            self._request_status()  # request one set of all messages on startup
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._request_status)
        self.status_timer.start(30000)  # 30000 milliseconds = 30 seconds

    def __del__(self):
        self.listener.stop()
        self.listener_thread.quit()
        self.listener_thread.wait()

    def handle_message(self, topic, contents):
        if topic not in ["ALIVE", "EXTERNALTRIGGER", "TRIGGERRATE"]:
            print("From Dastard: ", topic, contents)

        if topic == "ALIVE":
            running = contents.get("Running", False)
            if self.running != running:
                self.running = running

        elif topic == "STATUS":
            if self.running:
                source = contents.get("SourceName", "None")
                if self.source != source:
                    self.source = source
                projectors = contents.get("ChannelsWithProjectors", [])
                self.projectors = len(projectors) > 0
            self.status_updated.emit(contents)

        elif topic == "CHANNELNAMES":
            if self.channel_names != contents:
                self.channel_names = contents
                print(contents)

        elif topic == "WRITING":
            writing = contents.get("Active", False)
            if self.writing != writing:
                self.writing = writing
            if writing:
                filename = contents["FilenamePattern"] % ("chan1", "off")
                if self._off_filename != filename:
                    self.filename_changed.emit(filename)
                    self._off_filename = filename
            elif self._off_filename != "":
                self._off_filename = ""
                self.filename_changed.emit("")

        elif topic == "STATE":
            self.state_changed.emit(contents)

        elif topic == "TRIGGER":
            # print(contents)
            triggerList = [
                "AutoTrigger",
                "LevelTrigger",
                "EdgeTrigger",
                "EdgeMulti",
                "EdgeMultiNoise",
            ]
            for item in contents:
                if not any([item.get(trigger, False) for trigger in triggerList]):
                    if len(item["ChannelIndices"]) < self.get_n_channels():
                        self.disabled_chans = item["ChannelIndices"]
                        print(item["ChannelIndices"], "disabled")

    def _connect(self):
        try:
            self._socket = socket.create_connection(self.addr_port)
            self.connected = True
        except socket.error as ex:
            host, port = self.addr_port
            print(f"Could not connect to Dastard at {host}:{port}")
            self.connected = False
        return self.connected

    def _message(self, method_name, params):
        if not isinstance(params, list):
            params = [params]
        d = {"id": next(self._id_iter), "params": params, "method": method_name}
        return d

    def _call(self, method_name: str, params, verbose=True):
        if not self.connected:
            self._connect()
        if not self.connected:
            raise DastardError(
                "Not able to connect to Dastard, check running and try again"
            )

        msg = self._message(method_name, params)
        if verbose:
            if method_name == "SourceControl.ConfigureProjectorsBasis":
                # Params are waaaaay too long
                trunc_params = {"ChannelIndex": params["ChannelIndex"]}
                trunc_params["ProjectorsBase64"] = (
                    params["ProjectorsBase64"][:10]
                    + "..."
                    + params["ProjectorsBase64"][-10:]
                )
                trunc_params["BasisBase64"] = (
                    params["BasisBase64"][:10] + "..." + params["BasisBase64"][-10:]
                )
                trunc_msg = {
                    "id": msg["id"],
                    "method": msg["method"],
                    "params": [trunc_params],
                }
                trunc_str = json.dumps(trunc_msg)
                print(f"Dastard Client: sending {trunc_str}")
            else:
                print(f"Dastard Client: sending: {msg}")
        else:
            print(f"Dastard Client: calling {method_name}")
        self._socket.sendall(json.dumps(msg).encode())
        response = self._socket.recv(4096)

        if response == b"":
            print("Got b'', try reconnecting to Dastard")
            self._socket.close()
            self._connect()
            self._socket.sendall(json.dumps(msg).encode())
            response = self._socket.recv(4096)

        if response == b"":
            raise DastardError("no communication from Dastard")

        response = json.loads(response.decode())
        if verbose:
            print(f"Dastard Client: response: {response}")
        else:
            print(f"Dastard Client: got response for {method_name}")
        if not response["id"] == msg["id"]:
            raise DastardError("response id does not match message id")
        err = response.get("error", None)
        if err is not None:
            raise DastardError(f"""Dastard responded with error: {err}""")
        return response["result"]

    def _request_status(self):
        self._call("SourceControl.SendAllStatus", "dummy")

    def start_file(self, ljh22=None, off=None, path=None, filenamePattern=None):
        params = {"Request": "Start", "WriteLJH3": False}
        params.update(self.config.get("WriteControl", {}))

        if ljh22 is not None:
            params["WriteLJH22"] = ljh22
        if off is not None:
            params["WriteOFF"] = off
        if path is not None:
            params["Path"] = path
        if filenamePattern is not None:
            params["FilenamePattern"] = filenamePattern
        response = self._call("SourceControl.WriteControl", params)

        self.listener.update_messages()
        contents = self.listener.get_message_with_topic("WRITING")
        if not contents["Active"]:
            raise DastardError(
                f'Response from Dastard RPC should have contents["Active"]=True, but it does not\ncontents:\n{contents}'
            )
        self.off_filename = contents["FilenamePattern"] % ("chan1", "off")
        return self.off_filename

    def stop_source(self):
        response = self._call("SourceControl.Stop", "")
        return response

    def configure_simulate_pulse_source(
        self, nchan, sample_rate_hz, pedestal, amplitudes, samples_per_pulse
    ):
        params = {
            "Nchan": nchan,
            "SampleRate": sample_rate_hz,
            "Pedestal": pedestal,
            "Amplitudes": amplitudes,
            "Nsamp": samples_per_pulse,
        }
        response = self._call("SourceControl.ConfigureSimPulseSource", params)
        return response

    def start_sim_pulse_source(
        self,
        nchan=None,
        sample_rate_hz=None,
        pedestal=None,
        amplitudes=None,
        samples_per_pulse=None,
    ):
        params = {}
        params.update(self.config.get("simulation", {}))
        if nchan is not None:
            params["Nchan"] = nchan
        if sample_rate_hz is not None:
            params["SampleRate"] = sample_rate_hz
        if pedestal is not None:
            params["Pedestal"] = pedestal
        if amplitudes is not None:
            params["Amplitudes"] = amplitudes
        if samples_per_pulse is not None:
            params["Nsamp"] = samples_per_pulse

        response = self._call("SourceControl.ConfigureSimPulseSource", params)
        response = self._call("SourceControl.Start", "SIMPULSESOURCE")
        return response

    def set_experiment_state(self, state):
        params = {"Label": state, "WaitForError": True}
        response = self._call("SourceControl.SetExperimentStateLabel", params)
        return response

    def set_triggers(self, full_trigger_state):
        response = self._call("SourceControl.ConfigureTriggers", full_trigger_state)
        return response

    def stop_writing(self):
        params = {"Request": "Stop"}
        response = self._call("SourceControl.WriteControl", params)
        contents = self.listener.get_message_with_topic("WRITING")
        return contents

    def configure_record_lengths(self, npre=None, nsamp=None):
        params = self.config.get("pulseLengths", {})
        if nsamp is not None:
            params["Nsamp"] = nsamp
        if npre is not None:
            params["Npre"] = npre
        response = self._call("SourceControl.ConfigurePulseLengths", params)
        return response

    def get_data_path(self):
        return self._off_filename

    def set_projectors(self, projector_filename):
        source_type, _ = self.get_source_status()
        if source_type.lower() == "lancero":
            channels_per_pixel = 2
        else:
            channels_per_pixel = 1
        print(
            f"set_projectors found source_type={source_type} and therefore channels_per_pixel={channels_per_pixel}"
        )
        configs = getProjectorConfigs(
            projector_filename, self.get_name_to_number_index()
        )
        success_chans = []
        failures = OrderedDict()
        for channelIndex, config in list(configs.items()):
            # print("sending ProjectorsBasis for {}".format(channelIndex))
            try:
                response = self._call("SourceControl.ConfigureProjectorsBasis", config)
                success_chans.append(channelIndex)
            except DastardError as ex:
                failures[channelIndex] = repr(ex)

        success = len(failures) == 0
        result = (
            "success on channelIndices (not channelName): {}\n".format(
                sorted(success_chans)
            )
            + "failures:\n"
            + json.dumps(failures, sort_keys=True, indent=4)
        )
        print("set_projectors result")
        print(result)
        self._request_status()

    def get_source_status(self):
        return self.source, self.writing

    def get_n_channels(self):
        return len(self.get_name_to_number_index())

    def get_channel_indices(self):
        return list(self.get_name_to_number_index().values())

    # dastard channelNames go from chan1 to chanN and err1 to errN
    # we need to map from channelName to channelIndex (0-2N-1)
    def get_name_to_number_index(self):
        nameNumberToIndex = {}
        for i, name in enumerate(self.channel_names):
            if not name.startswith("chan"):
                continue
            nameNumber = int(name[4:])
            nameNumberToIndex[nameNumber] = i
            # for now since we only use this with lancero sources, error for non-odd index
            # if i % 2 != 1:
            #     raise Exception(
            #         "all fb channelIndices on a lancero source are odd, we shouldn't load projectors for even channelIndices")
        return nameNumberToIndex

    def set_pulse_trigger_all_chans(self, threshold=None, n_monotone=None):
        config = {"ChannelIndices": self.get_channel_indices()}
        config.update(self.config.get("pulseTrigger", {}))
        if threshold is not None:
            config["EdgeMultiLevel"] = threshold
        if n_monotone is not None:
            config["EdgeMultiVerifyNMonotone"] = n_monotone
        self._call("SourceControl.ConfigureTriggers", config)

        if self.config.get("use_bahama", False):
            self.configure_bahama_pulses()

    def set_noise_trigger_all_chans(self):
        config = {
            "ChannelIndices": self.get_channel_indices(),
        }
        config.update(self.config.get("noiseTrigger", {}))
        self._call("SourceControl.ConfigureTriggers", config)

        if self.config.get("use_bahama", False):
            self.configure_bahama_noise()

    def zero_all_triggers(self):
        config = {
            "ChannelIndices": self.get_channel_indices(),
        }
        self._call("SourceControl.ConfigureTriggers", config)

    def start_lancero(self):
        """
        Ported over from dc.py
        """
        mask = 0
        for k in range(16):
            mask |= 1 << k
        print("Fiber mask: 0x%4.4x" % mask)
        config = {"FiberMask": mask}
        config.update(self.config.get("lancero"))
        """
        clock = 125
        nsamp = 4

        activate = [0]
        delays = [1]

        for k, v in list(self.lanceroCheckBoxes.items()):
            if v.isChecked():
                activate.append(k)
                delays.append(self.lanceroDelays[k].value())

        chansep_columns = 0
        chansep_cards = 0
        firstrow = 1

        config = {
            "FiberMask": mask,
            "ClockMHz": clock,
            "CardDelay": delays,
            "Nsamp": nsamp,
            "FirstRow": firstrow,
            "ChanSepCards": chansep_cards,
            "ChanSepColumns": chansep_columns,
            "ActiveCards": activate,
            "AvailableCards": [],  # This is filled in only by server, not us.
        }
        """
        print("START LANCERO CONFIG")
        print(config)
        okay = self._call("SourceControl.ConfigureLanceroSource", config)
        if not okay:
            return False
        okay = self._call("SourceControl.Start", "LANCEROSOURCE")
        if not okay:
            return False
        return True

    def start_abaco(self):
        config = {"AvailableCards": []}
        config.update(self.config.get("abaco"))
        okay = self._call("SourceControl.ConfigureAbacoSource", config)
        if not okay:
            return False
        okay = self._call("SourceControl.Start", "ABACOSOURCE")
        if not okay:
            return False
        return True

    def start_source(self, source=None, restart=False):
        source = self.config.get("default_source", source)
        if source not in self.config["available_sources"]:
            raise ValueError(f"Requested source {source} not available")

        curr_source, running = self.get_source_status()
        if running:
            if curr_source.lower() == source:
                print(f"{source} already running")
                if restart:
                    self.stop_source()
                else:
                    return True
            else:
                self.stop_source()

        if source == "lancero":
            return self.start_lancero()
        elif source == "abaco":
            return self.start_abaco()
        elif source == "simulation":
            return self.start_sim_pulse_source()

    def configure_bahama_pulses(self, amplitudes=None, width=None, noiselevel=None):
        if amplitudes is None:
            amplitudes = self.config["bahama"].get("amplitudes", None)
        if width is None:
            width = self.config["bahama"].get("width", None)
        pulses = {}
        if amplitudes is not None:
            pulses["Amplitudes"] = amplitudes
        if width is not None:
            pulses["Width"] = width

        self.call_bahama("BahamaControl.ConfigurePulses", pulses)
        self.call_bahama(
            "BahamaControl.ConfigureBahama", {"Noiselevel": noiselevel, "Pulse": True}
        )
        self.call_bahama("BahamaControl.RegenerateData", {})

    def configure_bahama_noise(self, noiselevel=None, pulses=False):
        if noiselevel is None:
            noiselevel = self.config["bahama"].get("noiselevel", 0)
        self.call_bahama(
            "BahamaControl.ConfigureBahama", {"Noiselevel": noiselevel, "Pulse": False}
        )
        self.call_bahama("BahamaControl.RegenerateData", {})

    def call_bahama(self, method_name: str, params, verbose=True):
        if "bahama" not in self.config:
            raise ValueError("Bahama configuration not found in config")

        bahama_config = self.config["bahama"]
        host = bahama_config.get("host")
        port = bahama_config.get("port")

        if not host or not port:
            raise ValueError("Bahama host or port not specified in config")

        try:
            with socket.create_connection((host, port), timeout=5) as sock:
                msg = self._message(method_name, params)
                if verbose:
                    print(f"Bahama Client: sending: {msg}")
                else:
                    print(f"Bahama Client: calling {method_name}")

                sock.sendall(json.dumps(msg).encode())
                response = sock.recv(4096)

                if not response:
                    raise DastardError("No response from Bahama")

                response = json.loads(response.decode())

                if verbose:
                    print(f"Bahama Client: response: {response}")
                else:
                    print(f"Bahama Client: got response for {method_name}")

                if response.get("id") != msg["id"]:
                    raise DastardError("Response id does not match message id")

                err = response.get("error")
                if err is not None:
                    raise DastardError(f"Bahama responded with error: {err}")

                return response.get("result")

        except socket.error as e:
            raise DastardError(f"Socket error when communicating with Bahama: {e}")
        except json.JSONDecodeError as e:
            raise DastardError(f"Error decoding JSON response from Bahama: {e}")
        except Exception as e:
            raise DastardError(f"Unexpected error in Bahama communication: {e}")


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
    dt = np.dtype(
        [
            ("version", np.uint32),
            ("magic", np.uint8, (4,)),
            ("nrow", np.int64),
            ("ncol", np.int64),
            ("zeros", np.int64, 2),
            ("data", np.float64, nrow * ncol),
        ]
    )
    a = np.array(
        [(1, [ord("G"), ord("F"), ord("A"), 0], nrow, ncol, [0, 0], array.ravel())], dt
    )
    s_bytes = base64.b64encode(a)
    s = s_bytes.decode(encoding="ascii")
    return s, a[0]
