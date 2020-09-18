import numpy as np


def ssrl_10_1_mix_cal(cal_number, data):
    data.setDefaultBinsize(0.5)

    cal_state = f"CAL{cal_number}"
    ds = data.firstGoodChannel()
    ds.calibrationPlanInit("filtValue")
    ds.calibrationPlanAddPoint(4374, 'CKAlpha', states=cal_state)
    ds.calibrationPlanAddPoint(5992, 'NKAlpha', states=cal_state)
    ds.calibrationPlanAddPoint(7789, 'OKAlpha', states=cal_state)
    ds.calibrationPlanAddPoint(10047, 'FeLAlpha', states=cal_state)
    ds.calibrationPlanAddPoint(11730, 'NiLAlpha', states=cal_state)
    ds.calibrationPlanAddPoint(12599, 'CuLAlpha', states=cal_state)
    # ds.calibrationPlanAddPoint(13350, "mono", energy=1000)

    for ds in data.values()[1:]:
        ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)
    ds = data[1] # the above loop rebinds ds to the last dataset, but lets keep looking at the same one
    ds.learnResidualStdDevCut(n_sigma_equiv=10, plot=False, setDefault=True)

    data.alignToReferenceChannel(ds, "filtValue", np.arange(0, 30000,  6))
    data.calibrateFollowingPlan("filtValue", calibratedName="energy",
        dlo=15, dhi=15, overwriteRecipe=True)
    





# make sure this is at the bottom so all functions have been defined
routines_dict = {k:v for k,v in globals().items() if not k.startswith("__")}
def get(name: str):
    """get a routine by name string, the name matches the function name exactly"""
    assert name in routines_dict.keys(), f"routine {name} does not exist, existing routines are {list(routines_dict.keys())}"
    return routines_dict[name]