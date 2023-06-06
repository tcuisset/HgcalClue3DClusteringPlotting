import os
from functools import cached_property
import copy

import awkward as ak
import hist
import numpy as np
import pandas as pd
import uproot
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from ntupleReaders.computation import BaseComputation, computeAllFromTree
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from hists.dataframe import DataframeComputations
from hists.custom_hists import beamEnergiesAxis, layerAxis_custom
from hists.store import HistogramId
from hists.parameters import dEdx_weights
from HistogramLib.histogram import HistogramKind
from longitudinalProfile.violin import makeViolinBeamEnergy

from energy_resolution.sigma_over_e import fitSigmaOverEFromEnergyDistribution, EResolutionFitResult, SigmaOverEPlotElement, plotSCAsEllipse


class WeightedEnergyDistributionComputation(BaseComputation):
    neededBranches = ["beamEnergy", "rechits_layer", "rechits_energy"]

    def __init__(self, weights_dict:dict[int, float]) -> None:
        """ Parameters :
         - weights_dict : dict layer nb -> multiplicative factor to apply to hits energies on layer
        """
        self.weights_series = pd.Series(weights_dict).rename_axis("rechits_layer")
        self.h = hist.Hist(beamEnergiesAxis(), 
            hist.axis.Regular(bins=2000, start=0, stop=350, name="rechits_energy_sum_weighted", label="Rechits energy sum, with layer weights (GeV)"))

    def process(self, array: ak.Array) -> None:
        comp = DataframeComputations(array, rechits_columns=self.neededBranches)

        df = (
            comp.rechits_totalReconstructedEnergyPerEventLayer # Index : eventInternal, rechits_layer ; Columns: rechits_energy_sum_perLayer 
            .rechits_energy_sum_perLayer
            .multiply(self.weights_series, level=1) # Multiply by the weights for each layer
            .groupby("eventInternal").sum().rename("rechits_energy_sumPerEvent") # Sum energy per event

            .to_frame().join(comp.beamEnergy) # Add beamEnergy
        )
        self.h.fill(df.beamEnergy, df.rechits_energy_sumPerEvent)


class WeightedLongitudinalProfileComputation(BaseComputation):
    neededBranches = ["beamEnergy", "rechits_layer", "rechits_energy"]
    
    def __init__(self, weights_dict:dict[int, float]) -> None:
        """ Parameters :
         - weights_dict : dict layer nb -> multiplicative factor to apply to hits energies on layer
        """
        self.weights_series = pd.Series(weights_dict).rename_axis("rechits_layer")
        self.h = hist.Hist(beamEnergiesAxis(), 
            layerAxis_custom(name="rechits_layer", label="RecHit layer number"),
            storage=hist.storage.Mean())

    def process(self, array: ak.Array) -> None:
        comp = DataframeComputations(array, rechits_columns=self.neededBranches)
        df = comp.rechits_totalReconstructedEnergyPerEventLayer_allLayers()
        self.h.fill(df.beamEnergy, df.rechits_layer, sample=df.rechits_energy_sum_perLayer_fractionOfSynchrotronBeamEnergy)
   

def convert2DHistogramToDictOfHistograms(h:hist.Hist) -> dict[int, hist.Hist]:
    return {beamEnergy : h[{"beamEnergy" : hist.loc(beamEnergy)}] for beamEnergy in h.axes["beamEnergy"]}


def plotEllipsesComparedToDeDx(weightedFitResult:EResolutionFitResult, reader:ClueNtupleReader):
    dedx_rechits_plotElt = reader.loadSigmaOverEResults("rechits")

    weighted_plotElt = SigmaOverEPlotElement(legend="Layer weighted", fitResult=weightedFitResult, color="red")

    plotSCAsEllipse([dedx_rechits_plotElt, weighted_plotElt])



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
    def sigmaOverE(self) -> EResolutionFitResult:
        return fitSigmaOverEFromEnergyDistribution(convert2DHistogramToDictOfHistograms(self.weightedEnergyDistribution))
    
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
        fig, ax = plt.subplots()
        layers = list(self.weights.keys())
        corrected_dedx = [dEdx_weights[layer] * self.weights[layer] for layer in layers]
        original_dedx = [dEdx_weights[layer] for layer in layers]

        ax.plot(layers, original_dedx, "o", label="dEdx weights", color="blue", fillstyle="none")
        ax.plot(layers, corrected_dedx, "+", label="ML weights", color="orange")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer weights (MeV/MIP)")
        ax.xaxis.grid(True)

        hep.cms.text("Simulation Preliminary", ax=ax)
        hep.cms.lumitext("$e^+$ TB", ax=ax)
        ax.legend()
        return fig

    def plotEllipse(self):
        plotEllipsesComparedToDeDx(self.sigmaOverE, self.reader)
    
    def plotViolin(self):
        figsize = copy.copy(matplotlib.rcParams['figure.figsize'])
        figsize[0] *= 2
        fig, (ax_weighted, ax_unweighted) = plt.subplots(1, 2, figsize=figsize)

        makeViolinBeamEnergy(self.longitudinalProfile, self.reader.datatype, ax=ax_weighted)
        makeViolinBeamEnergy((self.reader.histStore
                .get(HistogramId("Clus3DClusteredEnergyPerLayer", self.reader.clueParams, self.reader.datatype))
                .getHistogram(HistogramKind.PROFILE)
                [{
                    "mainOrAllTracksters" : hist.loc("mainTrackster"),
                    # Project on clus3D_size
                    # keep beamEnergy, clus2D_layer
                }]
                .project("beamEnergy", "clus2D_layer")
            ), self.reader.datatype, ax=ax_unweighted)
        
        for ax, text in [(ax_weighted, "Layer weights from ML"), (ax_unweighted, "Layer weights from $\\frac{dE}{dx}$")]:
            ax.text(.99, .99, text, ha="right", va="top", transform=ax.transAxes)

        return fig
    
