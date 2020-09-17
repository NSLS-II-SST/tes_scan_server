import mass
import numpy as np
import h5py
import pylab as plt
import yaml

# this file will convert raw ljh files from ssrl_10_1 pre upgrade to off files for testing scan_server
# the ljh files are not included in the scan_server_repo for size
# they are found in two emails from Sang Jun, and the associated box.com links
#  https://stanford.box.com/s/kgf4wd6xevvv767q9zxbqurnh43sehk3
#  https://stanford.box.com/s/6jge81h033bzycajox7j32fz8cfla4yz


channums = [1,3,7,9,13,15,17,19]
pulse_basename = "20200219/20200219_016/20200219_016"
noise_basename = "20200219/20200219_014/20200219_014"
pulse_files = [pulse_basename+"_chan{}.ljh".format(channum) for channum in channums]
noise_files = [noise_basename+"_chan{}.ljh".format(channum) for channum in channums]

# this data was taken without zero threshold triggers so it won't work great
with h5py.File("ssrl_model.hdf5", "w") as h5:
    mass.make_projectors(
        pulse_files,
        noise_files,
        h5,
        n_sigma_pt_rms=1000,
        n_sigma_max_deriv=7,
        n_basis=5,
        maximum_n_pulses=3000,
        mass_hdf5_path=None,
        mass_hdf5_noise_path=None,
        invert_data=False,
        optimize_dp_dt=False,
        extra_n_basis_5lag=0,
        noise_weight_basis=True,
    )

models = {}
with h5py.File("ssrl_model.hdf5", "r") as h5:
    models = {int(ch) : mass.pulse_model.PulseModel.fromHDF5(h5[ch]) for ch in h5.keys()}
    models[1].plot()

ljhbases = [f"20200219/20200219_{n}/20200219_{n}" for n in ["016", "017", "018", "030"]]

ljh_filename_lists, off_filenames = mass.ljh2off.multi_ljh2off_loop(
    ljhbases,
    h5_path="ssrl_model.hdf5",
    off_basename="ssrl_10_1_scan/20200219",
    max_channels=200,
    n_ignore_presamples=2,
    require_experiment_state=False,
    show_progress=True,
)

off = mass.off.OffFile(off_filenames[0])
x,y = off.recordXY(0)

plt.figure()
plt.plot(x,y)


# now get the log info we need
with open("20200219/20200219_016/20200219_016_log","r") as f:
    d16 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open("20200219/20200219_017/20200219_017_log","r") as f:
    d17 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open("20200219/20200219_018/20200219_018_log","r") as f:
    d18 = list(yaml.load_all(f, Loader=yaml.SafeLoader))
with open("20200219/20200219_030/20200219_030_log","r") as f:
    d30 = list(yaml.load_all(f, Loader=yaml.SafeLoader))


noise_start = 1582150402.13
noise_end = d16[1]["header"]["start"]-1
cal0_start = d16[1]["header"]["start"]
cal0_stop = d16[1]["header"]["stop"]
cal1_start = d30[1]["header"]["start"]
cal1_stop = d30[1]["header"]["stop"]
