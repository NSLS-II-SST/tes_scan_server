from scan_server import DastardClient, DastardListener, DastardError
import time

i = 0
def sleep(x):
    global i
    print(f"start sleep {i} for {x}")
    time.sleep(x)
    i+=1

noise_trigger_state = {'ChannelIndices': [0], 'AutoTrigger': True, 'AutoDelay': 0, 'LevelTrigger': False, 'LevelRising': False, 'LevelLevel': 0, 'EdgeTrigger': False, 'EdgeRising': False, 'EdgeFalling': False, 'EdgeLevel': 0, 'EdgeMulti': False, 'EdgeMultiNoise': False, 'EdgeMultiMakeShortRecords': False, 'EdgeMultiMakeContaminatedRecords': False, 'EdgeMultiDisableZeroThreshold': False, 'EdgeMultiLevel': 0, 'EdgeMultiVerifyNMonotone': 0}
pulse_trigger_state = {'ChannelIndices': [0], 'AutoTrigger': False, 'AutoDelay': 0, 'LevelTrigger': False, 'LevelRising': False, 'LevelLevel': 0, 'EdgeTrigger': False, 'EdgeRising': False, 'EdgeFalling': False, 'EdgeLevel': 0, 'EdgeMulti': True, 'EdgeMultiNoise': False, 'EdgeMultiMakeShortRecords': False, 'EdgeMultiMakeContaminatedRecords': False, 'EdgeMultiDisableZeroThreshold': False, 'EdgeMultiLevel': 100, 'EdgeMultiVerifyNMonotone': 1}
print("starting")
host, port = ("192.168.1.143", 5500)
listener = DastardListener(host, port)
dc = DastardClient((host, port), listener)
try:
    dc.stop_source()
except DastardError as ex:
    print("source was already stopped")
    pass
dc.configure_simulate_pulse_source(1, sample_rate_hz=100000, pedestal=100, 
    amplitudes=[5000, 9000], samples_per_pulse=1000)
dc.start_sim_pulse_source()
dc.configure_record_lengths(npre=250, nsamp=500)
dc.set_triggers(pulse_trigger_state)
dc.start_writing_ljh22()
sleep(5)
dc.stop_writing()
# for now we have to stop the source to turn down the amplitudes
dc.stop_source() 
dc.configure_simulate_pulse_source(1, sample_rate_hz=100000, pedestal=100, 
    amplitudes=[0], samples_per_pulse=1000)
dc.start_sim_pulse_source()
dc.set_triggers(noise_trigger_state)
dc.start_writing_ljh22()
sleep(5)
dc.stop_writing()
# offline i copy the pulse files to this repo, then run
# python make_projectors data\0013\20210113_run0013_chan0.ljh data\0014\20210113_run0014_chan0.ljh -r
# to create projectors
dc.load_projectors("""data\0013\20210113_run0013_model.hdf5""")
dc.start_writing_off()
print("YO")