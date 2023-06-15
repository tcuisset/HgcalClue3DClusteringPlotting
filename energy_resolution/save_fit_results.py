
from collections import namedtuple
import pickle
import os

from hists.parameters import beamEnergies
from hist_loader import HistLoader
from sigma_over_e import SigmaOverEComputations, fitSigmaOverE, SigmaOverEPlotElement, sigmaOverE_fitFunction
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
#from fit import HistogramEstimates


def makeAndWriteFit(reader:ClueNtupleReader):
    loader = HistLoader(reader.histStore)
    comp_sigma_e = SigmaOverEComputations()
    os.makedirs(os.path.join(reader.pathToFolder, "sigmaOverE"), exist_ok=True)

    for level in ["rechits", "clue", "clue3d"]:
        sigma_e_results = comp_sigma_e.compute(
            {beamEnergy : loader.getProjected(reader.datatype, beamEnergy, level) for beamEnergy in beamEnergies}, 
            multiprocess="fork")
        
        sigmaOverE_fitResult = fitSigmaOverE(sigma_e_results)
        if level == "rechits":
            legend = "Rechits"
            color = "#1f77b4"
        elif level == "clue":
            legend = "CLUE"
            color = "#ff7f0e"
        elif level == "clue3d":
            legend = "CLUE3D"
            color = "#2ca02c"
        else:
            raise ValueError()
        
        if reader.datatype == "data":
            legend += " (Data)"
        else:
            legend += " (Simulation)" 
        plotElt = SigmaOverEPlotElement(
            legend=legend, fitResult=fitSigmaOverE(sigma_e_results), fitFunction=sigmaOverE_fitFunction,
            dataPoints={beamEnergy : result.sigma / result.mu for beamEnergy, result in sigma_e_results.items()},
            color=color)
        
        with open(os.path.join(reader.pathToSigmaOverEFolder, level + ".pickle"), 'wb') as f:
            pickle.dump(plotElt, f)

