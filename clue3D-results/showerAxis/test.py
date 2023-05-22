import sys
import math

import numpy as np
import hist
import hist.plot

sys.path.append("Plotting/")
from HistogramLib.histogram import HistogramKind
from HistogramLib.store import HistogramStore
from HistogramLib.iterative_fit import iterative_gaussian_fit
from hists.parameters import beamEnergies
from hists.store import HistogramId

hist_folder = '/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/v37'
#clueParams = "single-file"
clueParams = "cmssw"
histStore = HistogramStore(hist_folder, HistogramId)
datatypeToLegendMap = {"data":"Data", "sim_proton_v46_patchMIP":"Simulation"}
PCAMethodToLegendMap = {"filterLayerSpan":"No LC selection", "filterLayerSpan_cleaned":"Cleaned"}

def getHist(beamEnergy, datatype, angleType:str, PCA_method:str="filterLayerSpan"):
    if angleType == "angle":
        histName = "Clus3DAnglePCAToImpact"
        plotAxis = "clus3D_angle_pca_impact"
    else:
        histName = "Clus3DAnglePCAToImpact_XY"
        if angleType == "angle_x":
            plotAxis = "clus3D_angle_pca_impact_x"
        elif angleType == "angle_y":
            plotAxis = "clus3D_angle_pca_impact_y"
        else:
            raise ValueError()
    
    return (histStore.get(HistogramId(histName, clueParams, datatype))
        .getHistogram(HistogramKind.COUNT)[{
            "beamEnergy":hist.loc(beamEnergy),
            "mainOrAllTracksters":hist.loc("mainTrackster"),
            "PCA_method":hist.loc(PCA_method),
            # project on clus3D_size
        }]
    .project(plotAxis)
    )

w = iterative_gaussian_fit(getHist(20, "data", "angle_x"), startWindow=(-0.05, 0.05), sigmaRange=(-1.5, 1.3))