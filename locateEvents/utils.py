import numpy as np
import pandas as pd
from scipy.stats import binomtest
import hist


def makeDashLink(datatype, beamEnergy, ntuple, event, clueParam="cmssw"):
    return f"https://hgcal-tb18-clue3d-visualization.web.cern.ch/?clueParam={clueParam}&datatype={datatype}&beamEnergy={int(beamEnergy)}&ntuple={int(ntuple)}&event={int(event)}"

def makeCsvRow(beamEnergy, ntuple, event, *, source:str, layer:int=None):
    if layer is None:
        layerStr = ""
    else:
        layerStr = str(layer)
    return f"{int(beamEnergy)};{int(ntuple)};{int(event)};{layerStr};{source}"

def printCsvRowsFromDf(df:pd.DataFrame, source:str, layerColumn=None):
    colList = ["event", "ntupleNumber", "beamEnergy"]
    if layerColumn is not None:
        colList += [layerColumn]
    for row in df[colList].itertuples():
        if layerColumn is None:
            layer=None
        else:
            layer = getattr(row, layerColumn)
        print(makeCsvRow(row.beamEnergy, row.ntupleNumber, row.event, source=source, layer=layer))

def printDashLinksFromDf(df:pd.DataFrame, datatype, clueParam="cmssw"):
    for row in df[["beamEnergy", "event", "ntupleNumber"]].itertuples():
        print(makeDashLink(datatype, row.beamEnergy, row.ntupleNumber, row.event, clueParam))

def makeRatiosPerBeamEnergy(splitTracksters_hist:hist.Hist|np.ndarray, eventsPerBeamEnergy:hist.Hist|np.ndarray, alternative_hypothesis="two-sided"):
    """ Ratio of two histograms, returns tuple ratio_val, ratio_errors where : 
     - ratio_val is simply the ratio of splitTracksters_hist / eventsPerBeamEnergy
     - ratio_errors is an array (shape (2, N)) of confidence interval, binomial"""
    try:
        k_view = splitTracksters_hist.view()
    except:
        k_view = splitTracksters_hist
    try:
        n_view = eventsPerBeamEnergy.view().astype(int)
    except:
        n_view = eventsPerBeamEnergy
    
    ratio_val = []
    ratio_errors = []
    for i in range(len(n_view)):
        ratio = k_view[i]/n_view[i]
        low, high = binomtest(k=k_view[i], n=n_view[i], alternative=alternative_hypothesis).proportion_ci(0.95)
        ratio_val.append(ratio)
        ratio_errors.append((ratio-low, high-ratio))
    ratio_errors = np.transpose(np.array(ratio_errors))

    return ratio_val, ratio_errors