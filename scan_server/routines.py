def ssrl_10_1_mix_cal(cal_number, data):
    print(f"{cal_number=} {data=}")
    # ds = data.firstGoodChannel
    # ds.approxCal(lines=["CKAlpha", "OKAlpha", "FeLAlpha"])
    # data.alignToReferenceChannel(ds)
    # data.calibrateFollowingPlan()
    return data





# make sure this is at the bottom so all functions have been defined
routines_dict = {k:v for k,v in globals().items() if not k.startswith("__")}
def get(name: str):
    """get a routine by name string, the name matches the function name exactly"""
    assert name in routines_dict.keys(), f"routine {name} does not exist, existing routines are {list(routines_dict.keys())}"
    return routines_dict[name]