import numpy as np
import pickle
from mass.calibration import algorithms
from scan_server.mass_monkey_patch import unixnanos_to_state_slices

def ssrl_10_1_mix_cal(cal_number, data, attr, calibratedName):
    data.setDefaultBinsize(0.5)

    cal_state = f"CAL{cal_number}"
    ds = data.firstGoodChannel()
    line_names = ["CKAlpha", "NKAlpha", "OKAlpha", "FeLAlpha", "NiLAlpha", "CuLAlpha"]
    ds.learnCalibrationPlanFromEnergiesAndPeaks(attr=attr, states="CAL0", ph_fwhm=30, line_names=line_names)

    for ds in data.values()[1:]:
        ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)
    ds = data[1] # the above loop rebinds ds to the last dataset, but lets keep looking at the same one
    ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)

    data.alignToReferenceChannel(ds, attr, np.arange(0, 30000,  6))
    data.calibrateFollowingPlan(attr, calibratedName=calibratedName,
        dlo=15, dhi=15, overwriteRecipe=True)
    
def pkl_from_off(filename):
    i = filename.index('chan')
    pkl = filename[:i] + 'cal.pkl'
    return pkl

def simulated_cal_mix(cal_number, data, attr, calibratedName):
    data.setDefaultBinsize(0.5)

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


    
# make sure this is at the bottom so all functions have been defined
routines_dict = {k:v for k,v in globals().items() if not k.startswith("__")}
def get(name: str):
    """get a routine by name string, the name matches the function name exactly"""
    assert name in routines_dict.keys(), f"routine {name} does not exist, existing routines are {list(routines_dict.keys())}"
    return routines_dict[name]
