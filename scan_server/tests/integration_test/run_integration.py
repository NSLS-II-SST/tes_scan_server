from scan_server import DastardClient, DastardListener, DastardError
import time

import dastardcommander.projectors # no dependency for now since this integration test doesnt run on CI
from dastardcommander.projectors import toMatBase64
from collections import OrderedDict
import h5py

def getConfigs(filename):
    """
    returns an OrderedDict mapping channel number to a dict for use in calling
    self.client.call("SourceControl.ConfigureProjectorsBasis", config)
    to set Projectors and Bases
    extracts the channel numbers and projectors and basis from the h5 file
    filename - points to a _model.hdf5 file created by Pope
    """
    out = OrderedDict()
    h5 = h5py.File(filename, "r")
    for key in list(h5.keys()):
        nameNumber = int(key)
        channelIndex = (nameNumber)*2
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

i = 0
def sleep(x):
    global i
    print(f"start sleep {i} for {x}")
    time.sleep(x)
    i+=1

noise_trigger_state = {'ChannelIndices': [0], 'AutoTrigger': True, 'AutoDelay': 0, 'LevelTrigger': False, 'LevelRising': False, 'LevelLevel': 0, 'EdgeTrigger': False, 'EdgeRising': False, 'EdgeFalling': False, 'EdgeLevel': 0, 'EdgeMulti': False, 'EdgeMultiNoise': False, 'EdgeMultiMakeShortRecords': False, 'EdgeMultiMakeContaminatedRecords': False, 'EdgeMultiDisableZeroThreshold': False, 'EdgeMultiLevel': 0, 'EdgeMultiVerifyNMonotone': 0}
pulse_trigger_state = {'ChannelIndices': [0], 'AutoTrigger': False, 'AutoDelay': 0, 'LevelTrigger': False, 'LevelRising': False, 'LevelLevel': 0, 'EdgeTrigger': False, 'EdgeRising': False, 'EdgeFalling': False, 'EdgeLevel': 0, 'EdgeMulti': True, 'EdgeMultiNoise': False, 'EdgeMultiMakeShortRecords': False, 'EdgeMultiMakeContaminatedRecords': False, 'EdgeMultiDisableZeroThreshold': False, 'EdgeMultiLevel': 100, 'EdgeMultiVerifyNMonotone': 1}
host, port = ("192.168.1.143", 5500)
listener = DastardListener(host, port)
dc = DastardClient((host, port), listener)
def dastard_stop_source():
    try:
        dc.stop_source()
    except DastardError as ex:
        print("source was already stopped")
        pass

def make_dastard_write_files_from_which_we_can_create_projectors():
    dastard_stop_source()
    # set up simulated pulse source to make pulses with two different pulse heights
    dc.configure_simulate_pulse_source(1, sample_rate_hz=100000, pedestal=100, 
        amplitudes=[5000, 9000], samples_per_pulse=1000)
    dc.start_sim_pulse_source()
    dc.configure_record_lengths(npre=250, nsamp=500)
    dc.set_triggers(pulse_trigger_state)
    dc.start_writing(ljh22=True, off=False, path="/tmp")
    sleep(1)
    dc.stop_writing()
    # for now we have to stop the source to turn down the amplitudes
    # her we turn the amplitude to 0 so we can take "noise" data
    dc.stop_source() 
    dc.configure_simulate_pulse_source(1, sample_rate_hz=100000, pedestal=100, 
        amplitudes=[0], samples_per_pulse=1000)
    dc.start_sim_pulse_source()
    dc.set_triggers(noise_trigger_state)
    dc.start_writing(ljh22=True, off=False)
    sleep(1)
    dc.stop_writing()
    # offline i copied the pulse files to this repo, then run the mass make_projectors script with
    # python make_projectors data\0013\20210113_run0013_chan0.ljh data\0014\20210113_run0014_chan0.ljh -r
    # to create projectors


def make_dastard_be_writing_off_files(start_writing):
    dastard_stop_source()
    dc.configure_simulate_pulse_source(1, sample_rate_hz=100000, pedestal=100, 
        amplitudes=[5000, 9000], samples_per_pulse=1000)
    dc.start_sim_pulse_source()
    dc.configure_record_lengths(npre=250, nsamp=500)    
    dc.set_triggers(pulse_trigger_state)
    configs = getConfigs("""data/0013/20210113_run0013_model.hdf5""")
    print(f"{configs.keys()=}")
    print(f"{configs[0]=}")
    for channelIndex, config in list(configs.items()):
        print("sending ProjectorsBasis for {}".format(channelIndex))
        dc._call("SourceControl.ConfigureProjectorsBasis", config, verbose=False)
    # at this point dastard is ready for TESScanner to be in state "no_file"
    if start_writing:
        # TESScanner could be made consistent with "file_open" with some work
        dc.start_writing(ljh22=False, off=True, path="/tmp")

# make_dastard_write_files_from_which_we_can_create_projectors()
make_dastard_be_writing_off_files(start_writing=True)

# to get further towards a beamline free integration test
# we should change start_writing to False above
# then instantiate a TESScanner and simulate some scans

#then could insantiate a server, and simulate scans from an external machine