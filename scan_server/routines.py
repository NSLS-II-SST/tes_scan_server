import numpy as np
import pickle
from mass.calibration import algorithms
import mass
from mass.off import ChannelGroup, getOffFileListFromOneFile
from scan_server.mass_monkey_patch import unixnanos_to_state_slices
import argparse

def run_cal_routine():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to an off file that we will cal")
    parser.add_argument("states", type=int, help="One state number to cal")
    parser.add_argument("--attr", default="filtValue")
    parser.add_argument("--calibratedName", default="energy")
    args = parser.parse_args()
    data = ChannelGroup(getOffFileListFromOneFile(args.filename))
    nsls_mix_980eV(args.states, data, args.attr, args.calibratedName)
    

def nsls_mix_980eV(cal_number, data, attr, calibratedName):
    data.setDefaultBinsize(0.2)
    cal_state = f"CAL{cal_number}"
    ds = data.firstGoodChannel()
    ds.learnCalibrationPlanFromEnergiesAndPeaks(attr=attr, 
                                                ph_fwhm=50, states=cal_state,
                                                line_names=["CKAlpha", "NKAlpha",
                                                            "OKAlpha", "FeLAlpha",
                                                            "NiLAlpha", 'CuLAlpha'],
                                                maxacc=0.1)
    ds.calibrateFollowingPlan(attr, overwriteRecipe=True, dlo=20, dhi=25)
    # ds.diagnoseCalibration()
    data.alignToReferenceChannel(ds, attr, np.arange(0, 20000,  10))
    data.calibrateFollowingPlan(attr, calibratedName,
        dlo=20, dhi=25, overwriteRecipe=True)
    data.calibrationSaveToHDF5Simple("/home/xf07id1/.scan_server/nsls_server_saved_calibration.hdf5")
    success = True
    return success

def simulated_source(cal_number, data, attr, calibratedName):
    data.setDefaultBinsize(0.1)

    cal_state = f"CAL{cal_number}"
    ds = data.firstGoodChannel()
    line_names = ["600", "800", "1000"] # dcom hard codes 0.6, 0.8, 1.0 ratios for pulses
    mass.STANDARD_FEATURES["600"]=600
    mass.STANDARD_FEATURES["800"]=800
    mass.STANDARD_FEATURES["1000"]=1000 # dcom hard codes 0.6, 0.8, 1.0 ratios for pulses
    ds.learnCalibrationPlanFromEnergiesAndPeaks(attr=attr, states=cal_state, ph_fwhm=30, line_names=line_names)

    # for ds in data.values()[1:]:
    #     ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)
    # ds = data[1] # the above loop rebinds ds to the last dataset, but lets keep looking at the same one
    # ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)

    data.alignToReferenceChannel(ds, attr, np.arange(0, 5000,  1))
    data.calibrateFollowingPlan(attr, calibratedName=calibratedName,
        dlo=15, dhi=15, overwriteRecipe=True)
    success = True
    return success

def ssrl_10_1_mix_cal(cal_number, data, attr, calibratedName):
    data.setDefaultBinsize(0.5)

    cal_state = f"CAL{cal_number}"
    ds = data.firstGoodChannel()
    line_names = ["CKAlpha", "NKAlpha", "OKAlpha", "FeLAlpha", "NiLAlpha", "CuLAlpha"]
    ds.learnCalibrationPlanFromEnergiesAndPeaks(attr=attr, states=cal_state, ph_fwhm=30, line_names=line_names, maxacc=0.015)

    for ds in data.values()[1:]:
        ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)
    ds = data[1] # the above loop rebinds ds to the last dataset, but lets keep looking at the same one
    ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)

    data.alignToReferenceChannel(ds, attr, np.arange(0, 30000,  6))
    data.calibrateFollowingPlan(attr, calibratedName=calibratedName,
        dlo=15, dhi=15, overwriteRecipe=True)
    success = True
    return success

def pkl_from_off(filename):
    i = filename.index('chan')
    pkl = filename[:i] + 'cal.pkl'
    return pkl

def simulated_cal_mix(cal_number, data, attr, calibratedName):
    data.setDefaultBinsize(0.5)
    mass.line_models.VALIDATE_BIN_SIZE = False
    cal_state = f"CAL{cal_number}"
    ds = data.firstGoodChannel()
    line_names = ['Line1', 'Line2']
    peak_locations, peak_intensities = algorithms.find_local_maxima(ds.getAttr('filtValue', indsOrStates=[cal_state]), 10)
    ph_raw = peak_locations[:2]
    ph_raw.sort()
    cal_energies = [200, 400]
    ds.calibrationPlanInit(attr)
    for ph, name, e in zip(ph_raw, line_names, cal_energies):
        ds.calibrationPlanAddPoint(ph, name, states=[cal_state], energy=e)

    filename = ds.offFile.filename
    cal_pkl = pkl_from_off(filename)
    cal_dict = {ds.channum: {"ph_raw": ph_raw, "cal_energies": cal_energies, "line_names": line_names, "attr": attr}}
    with open(cal_pkl, 'wb') as f:
        pickle.dump(cal_dict, f)

    data.alignToReferenceChannel(ds, attr, np.arange(0, 10000,  6))
    data.calibrateFollowingPlan(attr, calibratedName=calibratedName,
        dlo=15, dhi=15, overwriteRecipe=True)

    nchan = len(data.values())
    nbad = len(data.whyChanBad)
    badRatio = nbad/nchan
    success = (badRatio < 0.75)
    return success


    
# make sure this is at the bottom so all functions have been defined
routines_dict = {k:v for k,v in globals().items() if not k.startswith("__")}
def get(name: str):
    """get a routine by name string, the name matches the function name exactly"""
    assert name in routines_dict.keys(), f"routine {name} does not exist, existing routines are {list(routines_dict.keys())}"
    return routines_dict[name]
