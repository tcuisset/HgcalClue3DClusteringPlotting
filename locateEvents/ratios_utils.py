import numpy as np
import pandas as pd
from scipy.stats import binomtest
import hist

from HistogramLib.store import HistogramStore
from HistogramLib.histogram import HistogramKind
from hists.store import HistogramId
from hists.parameters import synchrotronBeamEnergiesMap, beamEnergies


def makeFrequencyPerBeamEnergy(histStore:HistogramStore, clueParams, series:pd.Series, datatype, alternative_hypothesis="two-sided"):
    """ Make ratio of frequency of occurence per beam energy
    Parameters : 
     - series : a Pandas series of beam energy values to histogram (or anything that can be passed to hist.Hist.fill)
    """
    eventsPerBeamEnergy = histStore.get(HistogramId("EventsPerBeamEnergy", clueParams, datatype)).getHistogram(HistogramKind.COUNT)
    beamEnergy_axis = eventsPerBeamEnergy.axes[0]
    splitTracksters_hist = hist.Hist(beamEnergy_axis, storage=hist.storage.Int64())
    splitTracksters_hist.fill(series)
    k_view = splitTracksters_hist.view()
    n_view = eventsPerBeamEnergy.view().astype(int)
    ratio_val = []
    ratio_errors = []
    for i in range(len(n_view)):
        ratio = k_view[i]/n_view[i]
        low, high = binomtest(k=k_view[i], n=n_view[i], alternative=alternative_hypothesis).proportion_ci(0.95)
        ratio_val.append(ratio)
        ratio_errors.append((ratio-low, high-ratio))
    ratio_errors = np.transpose(np.array(ratio_errors))

    return ratio_val, ratio_errors

