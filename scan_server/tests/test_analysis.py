from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
from scan_server import TESScanner, Scan, DastardClient
import mass
import os
import tempfile
import yaml
import numpy as np
import pylab as plt
from scan_server.tests import util


def test_analysis():


    d16, d17, d18, d30 = util.scan_logs_raw()


    # write a dummy experiment state file, since the data didn't come with one
    filename = tempfile.mktemp()
    util.write_ssrl_experiment_state_file(filename)

    experiment_state_file = mass.off.channels.ExperimentStateFile(filename)

    data = ChannelGroup(getOffFileListFromOneFile(os.path.join(util.ssrl_dir, "20200219_chan1.off"), maxChans=3),
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


    results = [scan.hist2d(data, np.arange(0, 1000, 1), "energy") for scan in util.scans()]

    results[0].plot()
    results[1].plot()
    mega_result = results[0] + results[1] + results[2] + results[3] + results[4] + results[5] 
    mega_result.plot()