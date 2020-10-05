import mass
import numpy as np
from mass.calibration import algorithms


def unixnanos_to_state_slices(ds_unixnano, starts_unixnano, ends_unixnano):
    starts_ind = np.searchsorted(ds_unixnano, starts_unixnano)
    ends_ind = np.searchsorted(ds_unixnano, ends_unixnano)
    states = [slice(a, b, None) for a, b in zip(starts_ind, ends_ind)]
    return states    

def ds_unixnanoToStates(self, starts_unixnano, ends_unixnano):
    return unixnanos_to_state_slices(self.unixnano, starts_unixnano, ends_unixnano)


mass.off.Channel.unixnanoToStates = ds_unixnanoToStates

def data_histWithUnixnanos(self, bin_edges, attr, starts_unixnano, ends_unixnano):
    counts = np.zeros(len(bin_edges)-1, dtype="int64")
    for ds in self.values():
        states = ds.unixnanoToStates(starts_unixnano, ends_unixnano)
        bin_centers, _counts = ds.hist(bin_edges, attr, states=states)
        counts += _counts
    return bin_centers, counts

mass.off.ChannelGroup.histWithUnixnanos = data_histWithUnixnanos

def ds_learnCalibrationPlanFromEnergiesAndPeaks(self, attr, states, ph_fwhm, line_names):
    peak_ph_vals, _peak_heights = algorithms.find_local_maxima(self.getAttr(attr, indsOrStates=states), ph_fwhm)
    _name_e, energies_out, opt_assignments = algorithms.find_opt_assignment(peak_ph_vals, line_names)

    self.calibrationPlanInit(attr)
    for ph, name in zip(opt_assignments, _name_e):
        self.calibrationPlanAddPoint(ph, name, states=states)
mass.off.Channel.learnCalibrationPlanFromEnergiesAndPeaks = ds_learnCalibrationPlanFromEnergiesAndPeaks

# from collections import OrderedDict
# from typing import List, Dict


# class ExperimentStateNoBacktrack():
#     """like and mass.off.ExperimentStateFile, but no file involved, and you can't make a state until
#     you know when it is finished
#     also doesn't support aliasing"""
#     def __init__(self):
#         self.stateDescriptions: Dict[List[slice]] = OrderedDict() # slices of unixnanos
#         #meaningless items neccesary for ChannelGroup.refreshFromFiles
#         self.allLabels = []
#         self.parse = lambda: None

#     def calcStateDictFull(self, ds_unixnano) -> dict:
#         out = OrderedDict()
#         for k, slices_unixnano in self.stateDescriptions.items():
#             starts = [s.start for s in slices_unixnano]
#             ends = [s.stop for s in slices_unixnano]
#             slices_inds = unixnanos_to_state_slices(ds_unixnano, starts, ends)
#             out[k] = slices_inds
#         return out

#     def calcStatesDict(self, ds_unixnano, statesDict=None, i0_allLabels=0, i0_unixnanos=0):
#         return self.calcStateDictFull(ds_unixnano)

#     def addState(self, statename, a, b):
#         descriptions = self.stateDescriptions.get(statename, [])
#         descriptions.append(slice(a, b, None))

#     @property
#     def labels(self):
#         return list(self.stateDescriptions.keys())


    

