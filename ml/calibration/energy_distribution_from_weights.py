import os
from functools import cached_property
import copy
import typing

import awkward as ak
import hist
import numpy as np
import pandas as pd
import uproot
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from ntupleReaders.computation import BaseComputation, computeAllFromTree, Report, ComputationToolBase
from ntupleReaders.tools import DataframeComputationsToolMaker
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from hists.dataframe import DataframeComputations
from hists.custom_hists import beamEnergiesAxis, layerAxis_custom
from hists.store import HistogramId
from hists.parameters import dEdx_weights
from HistogramLib.histogram import HistogramKind
from longitudinalProfile.violin import makeViolinBeamEnergy

from energy_resolution.sigma_over_e import SigmaOverEComputations, fitSigmaOverEFromEnergyDistribution, EResolutionFitResult, SigmaOverEPlotElement, plotSCAsEllipse, fitSigmaOverE

class WeightedEnergyDistributionComputation(BaseComputation):
    """ Computes an histogram of the total energy distribution, computed using the given layer weights, for each beam energy """

    def __init__(self, weights_dict:dict[int, float]) -> None:
        """ Parameters :
         - weights_dict : dict layer nb -> multiplicative factor to apply to hits energies on layer
        """
        super().__init__(neededBranches=["beamEnergy", "rechits_layer", "rechits_energy"], 
            neededComputationTools=[DataframeComputationsToolMaker(rechits_columns=["rechits_layer", "rechits_energy"])])
        
        self.weights_series = pd.Series(weights_dict).rename_axis("rechits_layer")
        self.h = hist.Hist(beamEnergiesAxis(), 
            hist.axis.Regular(bins=2000, start=0, stop=350, name="rechits_energy_sum_weighted", label="Rechits energy sum, with layer weights (GeV)"))

    def processBatch(self, array:ak.Array, report:Report, computationTools:dict[typing.Type[ComputationToolBase], ComputationToolBase]) -> None:
        comp = computationTools[DataframeComputations]

        df = (
            comp.rechits_totalReconstructedEnergyPerEventLayer # Index : eventInternal, rechits_layer ; Columns: rechits_energy_sum_perLayer 
            .rechits_energy_sum_perLayer
            .multiply(self.weights_series, level="rechits_layer") # Multiply by the weights for each layer
            .groupby("eventInternal").sum().rename("rechits_energy_sumPerEvent") # Sum energy per event

            .to_frame().join(comp.beamEnergy) # Add beamEnergy
        )
        self.h.fill(df.beamEnergy, df.rechits_energy_sumPerEvent)


class WeightedLongitudinalProfileComputation(BaseComputation):
    """ Compute an histogram of the """
    
    def __init__(self, weights_dict:dict[int, float]) -> None:
        """ Parameters :
         - weights_dict : dict layer nb -> multiplicative factor to apply to hits energies on layer
        """
        super().__init__(neededBranches=["beamEnergy", "rechits_layer", "rechits_energy"], 
            neededComputationTools=[DataframeComputationsToolMaker(rechits_columns=["rechits_layer", "rechits_energy"])])
        
        self.weights_series = pd.Series(weights_dict).rename_axis("rechits_layer")
        self.h = hist.Hist(beamEnergiesAxis(), 
            layerAxis_custom(name="rechits_layer", label="RecHit layer number"),
            storage=hist.storage.Mean())

    def processBatch(self, array:ak.Array, report:Report, computationTools:dict[typing.Type[ComputationToolBase], ComputationToolBase]) -> None:
        comp = computationTools[DataframeComputations]
        df = comp.rechits_totalReconstructedEnergyPerEventLayer_allLayers(reset_layer_index=False)
        self.h.fill(df.beamEnergy, df.reset_index("rechits_layer").rechits_layer, sample=df.rechits_energy_sum_perLayer.multiply(self.weights_series, level="rechits_layer"))
   

# def plotEllipsesComparedToDeDx(weightedFitResult:EResolutionFitResult, reader:ClueNtupleReader):
#     dedx_rechits_plotElt = reader.loadSigmaOverEResults("rechits")

#     weighted_plotElt = SigmaOverEPlotElement(legend="Layer weighted", fitResult=weightedFitResult, color="red")

#     plotSCAsEllipse([dedx_rechits_plotElt, weighted_plotElt])
#     hep.cms.text("Preliminary")
#     hep.cms.lumitext("$e^+$ TB")


class WeightedLayersComputations:
    def __init__(self, weights:dict[int, float], reader:ClueNtupleReader) -> None:
        """
        Parameters :
         - weights : weights per layer, multiplicative (dict layer nb -> weight)
        """
        self.weights = weights
        self.reader = reader
        self.computeEnergyDistribution = True
        self.computeLongitudinalProfile = True
    
    def compute(self):
        computations = []
        if self.computeEnergyDistribution:
            self.weightedEnergyDistributionComp = WeightedEnergyDistributionComputation(self.weights)
            computations.append(self.weightedEnergyDistributionComp)

        if self.computeLongitudinalProfile:
            self.longitudinalProfileComp = WeightedLongitudinalProfileComputation(self.weights)
            computations.append(self.longitudinalProfileComp)
        
        computeAllFromTree(self.reader.tree, computations, tqdm_options=dict(desc="Histogramming with weighted layers"))

    @property
    def weightedEnergyDistribution(self) -> hist.Hist:
        """ Returns : a histogram with axes :
        - beamEnergy
        - rechits_energy_sum_weighted : the distribution of the sum of rechits energies, using the layer weights
        """
        return self.weightedEnergyDistributionComp.h
    
    @property
    def longitudinalProfile(self) -> hist.Hist:
        """ Longitudinale profile of showers.
        Axes : 
         - beamEnergy
         - layer
         - mean sum of energy on layer, divided by beam energy (incl synchrotron radiation correction)
        """
        return self.longitudinalProfileComp.h
    
    @cached_property
    def sigmaOverEComputation(self) -> SigmaOverEComputations:
        comp = SigmaOverEComputations()
        comp.compute(self.weightedEnergyDistribution, multiprocess="forkserver")
        return comp
    
    @property
    def sigmaOverE(self) -> EResolutionFitResult:
        return fitSigmaOverE(self.sigmaOverEComputation.results)
    
    def plotRelativeWeights(self):
        fig, ax = plt.subplots()
        ax.plot(self.weights.keys(), self.weights.values(), "+")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Weights (multiplicative to dE/dx, dimensionless)")
        ax.xaxis.grid(True)

        hep.cms.text("Simulation Preliminary", ax=ax)
        hep.cms.lumitext("$e^+$ TB", ax=ax)
        return fig

    def plotDeDxWeights(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        layers = list(self.weights.keys())
        corrected_dedx = [dEdx_weights[layer] * self.weights[layer] for layer in layers]
        original_dedx = [dEdx_weights[layer] for layer in layers]

        common_kwargs = dict(markersize=15, markeredgewidth=2)
        ax.plot(layers, original_dedx, "o", label=r"$\frac{dE}{dx}$ weights", color="tab:blue", fillstyle="none", **common_kwargs)
        ax.plot(layers, corrected_dedx, "+", label="Fitted weights", color="tab:orange", **common_kwargs)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer weights (MeV/MIP)")
        ax.xaxis.grid(True)

        #hep.cms.text("Simulation Preliminary", ax=ax)
        hep.cms.lumitext("Calibrated layer weights - $e^+$ Test Beam", ax=ax)
        ax.legend(frameon=True)
        return fig

    # def plotEllipse(self):
    #     plotEllipsesComparedToDeDx(self.sigmaOverE, self.reader)
    
    def plotViolin(self):
        figsize = copy.copy(matplotlib.rcParams['figure.figsize'])
        figsize[0] *= 2
        fig, (ax_weighted, ax_unweighted) = plt.subplots(1, 2, figsize=figsize)

        makeViolinBeamEnergy(self.longitudinalProfile, self.reader.datatype, ax=ax_weighted)
        makeViolinBeamEnergy((self.reader.histStore
                .get(HistogramId("RechitsEnergyReconstructedPerLayer", self.reader.clueParams, self.reader.datatype))
                .getHistogram(HistogramKind.PROFILE)
                #.project("beamEnergy", "rechits_layer")
            ), self.reader.datatype, ax=ax_unweighted)
        
        for ax, text in [(ax_weighted, "Layer weights from ML"), (ax_unweighted, "Layer weights from $\\frac{dE}{dx}$")]:
            ax.text(.99, .99, text, ha="right", va="top", transform=ax.transAxes)

        return fig
    
