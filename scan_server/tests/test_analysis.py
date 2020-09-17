from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
from scan_server import TESScanner, Scan, DastardClient
import mass
import os
import tempfile
import yaml
import numpy as np
import pylab as plt
from scan_server.tests import util






d16, d17, d18, d30 = util.scan_logs_raw()


# write a dummy experiment state file, since the data didn't come with one
with open(tempfile.mktemp(), "w") as f:
    f.write("# yo yo\n")
    f.write("0, START\n")
    cal_start, cal_stop = int(d16[1]["header"]["start"]*1e9), int(d16[1]["header"]["stop"]*1e9)
    f.write(f"{cal_start}, CAL0\n")
    f.write(f"{cal_stop}, PAUSE\n")
    for j, scan in enumerate(util.scans()):
        f.write(f"{int(scan.epoch_time_start_s[0]*1e9)}, SCAN{j}\n")
        f.write(f"{int(scan.epoch_time_end_s[-1]*1e9)}, PAUSE\n")
        # for i, mono_val in enumerate(scan0_mono_vals):
        #     start, end = scan_log["mono"][mono_val]
        #     f.write(f"{int(start*1e9)}, SCAN{j}_POINT{i}\n")
        #     f.write(f"{int(end*1e9)}, PAUSE\n")
    cal_start, cal_stop = int(d30[1]["header"]["start"]*1e9), int(d30[1]["header"]["stop"]*1e9)
    f.write(f"{cal_start}, CAL1\n")
    f.write(f"{cal_stop}, PAUSE\n")
experiment_state_file = mass.off.channels.ExperimentStateFile(f.name)

data = ChannelGroup(getOffFileListFromOneFile(os.path.join(util.ssrl_dir, "20200219_chan1.off"), maxChans=8),
        experimentStateFile=experiment_state_file)
data.setDefaultBinsize(0.5)

ds = data.firstGoodChannel()
ds.calibrationPlanInit("filtValue")
ds.calibrationPlanAddPoint(4374, 'CKAlpha')
ds.calibrationPlanAddPoint(5992, 'NKAlpha')
ds.calibrationPlanAddPoint(7789, 'OKAlpha')
ds.calibrationPlanAddPoint(10047, 'FeLAlpha')
ds.calibrationPlanAddPoint(11730, 'NiLAlpha')
ds.calibrationPlanAddPoint(12599, 'CuLAlpha')
# ds.calibrationPlanAddPoint(13350, "mono", energy=1000)

for ds in data.values()[1:]:
    ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)
ds = data[1] # the above loop rebinds ds to the last dataset, but lets keep looking at the same one
ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)

data.alignToReferenceChannel(ds, "filtValue", np.arange(0, 30000,  6))
data.calibrateFollowingPlan("filtValue", calibratedName="energy",
    dlo=15, dhi=15, overwriteRecipe=True)
# ds.diagnoseCalibration()
# 

from dataclasses import dataclass
from typing import Any

@dataclass
class ScanResult():
    hist2d: Any # really np.ndarray
    var_name: str
    var_unit: str
    var_vals: Any # really np.nda
    bin_centers: Any # really np.ndarray
    attr: str
    attr_unit: str
    scan_desc: str

    def plot(self):
        plt.figure()
        plt.contourf(self.var_vals, self.bin_centers, self.hist2d, cmap="gist_heat")
        plt.xlabel(f"{self.var_name} ({self.var_unit})")
        plt.ylabel(f"{self.attr} ({self.attr_unit})")
        plt.title(self.scan_desc)

    def __add__(self, other):
        assert self.var_name == other.var_name
        assert self.var_unit == other.var_unit
        assert all([a == b for (a, b) in zip(self.var_vals, other.var_vals)])
        assert all([a == b for (a, b) in zip(self.bin_centers, other.bin_centers)])
        assert self.attr == other.attr
        assert self.attr_unit == other.attr_unit
        return ScanResult(self.hist2d+other.hist2d, self.var_name, self.var_unit, self.var_vals,
        self.bin_centers, self.attr, self.attr_unit, "sum scan")

def scan_2d_hist(scan, data, bin_edges, attr):
    starts_nano = (1e9*np.array(scan.epoch_time_start_s)).astype(int)
    ends_nano = (1e9*np.array(scan.epoch_time_end_s)).astype(int)
    assert len(starts_nano) == len(ends_nano)
    var_name = scan.var_name
    var_unit = scan.var_unit
    var_vals = scan.var_values
    hist2d = np.zeros((len(bin_edges)-1, len(starts_nano)))
    for ds in data.values():
        ind_starts = np.searchsorted(ds.unixnano, starts_nano)
        ind_ends = np.searchsorted(ds.unixnano, ends_nano)
        for i, (a, b) in enumerate(zip(ind_starts, ind_ends)):
            states = slice(a, b, None)
            bin_centers, counts = ds.hist(bin_edges, attr, states=states)
            hist2d[:, i] += counts
    return ScanResult(hist2d, var_name, var_unit, var_vals, bin_centers, attr, "eV", scan.description_str())
results = [scan_2d_hist(scan, data, np.arange(0, 1000, 1), "energy") for scan in util.scans()]

result[0].plot()
sum(results).plot()