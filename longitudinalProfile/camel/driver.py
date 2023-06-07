import os
from functools import cached_property

import uproot
import numpy as np
import pandas as pd
import hist
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

from event_visualizer.event_index import EventLoader
from event_visualizer.notebook_visualizer import EventDisplay
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from ntupleReaders.computation import computeAllFromTree, BaseComputation, ComputationToolMakerBase, NoFilter
from HistogramLib.store import HistogramStore
from HistogramLib.histogram import HistogramKind
from hists.custom_hists import beamEnergiesAxis
from hists.store import HistogramId

from longitudinalProfile.camel.peak_finding import MakePeaksDfComputation
import longitudinalProfile.camel.filtering as filtering
from longitudinalProfile.camel.filtering import FilterLowBaseHeight, FilterNoisyCellsPeaks, FilterDipsNotInCenter, FilterApplier, FilterNoisyCellsPeaksUsingBases
from longitudinalProfile.camel.plot import plotIndividualProfile_scipy

class EventList:
    def __init__(self, peaks_df_subset:pd.DataFrame, energyPerLayer_df:pd.DataFrame, eventLoader:EventLoader) -> None:
        self.peaks_df_subset = peaks_df_subset
        self.energyPerLayer_df = energyPerLayer_df
        self.eventLoader = eventLoader
        self.cur_index_i = 0
        self.cur_eventInternal = self.peaks_df_subset.index[0]
    
    def query(self, query_str:str):
        return EventList(self.peaks_df_subset.query(query_str), self.energyPerLayer_df, self.eventLoader)
    
    def __len__(self):
        return len(self.peaks_df_subset)
    
    def next(self):
        self.cur_index_i += 1
        self.cur_eventInternal = self.peaks_df_subset.index[self.cur_index_i]
    
    def setIndex(self, index):
        self.cur_index_i = index
        self.cur_eventInternal = self.peaks_df_subset.index[self.cur_index_i]

    def _getCurrentEvent(self):
        try:
            peaksInfo = self.peaks_df_subset.peaks_info.loc[self.cur_eventInternal]
        except:
            # Make a peaksInfo object with a single peak
            peaksInfo = np.array([self.peaks_df_subset.peak_index.loc[self.cur_eventInternal]]), {key : np.array([self.peaks_df_subset[key].loc[self.cur_eventInternal]]) for key in ["prominences","left_bases","right_bases","widths","width_heights","left_ips","right_ips"]}
        
        out_dict = dict(peaksInfo=peaksInfo, energyPerLayer=self.energyPerLayer_df.rechits_energy_sum_perLayer.loc[self.cur_eventInternal])
        try:
            out_dict["beamEnergy"] = self.peaks_df_subset.beamEnergy.loc[self.cur_eventInternal]
            out_dict["ntupleNumber"] = self.peaks_df_subset.ntupleNumber.loc[self.cur_eventInternal]
            out_dict["event"] = self.peaks_df_subset.event.loc[self.cur_eventInternal]
        except:
            pass
        return out_dict
    
    def plotLongitudinalProfile(self, ax=None):
        event_data = self._getCurrentEvent()
        plotIndividualProfile_scipy(event_data["energyPerLayer"], *event_data["peaksInfo"], ax=ax)
        hep.cms.text("Preliminary", loc=2)
        
        if ax is None:
            ax = plt.gca()
        plt.ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
        try:
            #hep.cms.label("Preliminary")
            text = f'{event_data["beamEnergy"]:.0f} GeV - ntuple {event_data["ntupleNumber"]} - event {event_data["event"]}'
            #hep.cms.lumitext(text)
            ax.text(x=0.99, y=0.95, s=text, transform=ax.transAxes, ha="right", va="bottom", fontsize=20)
        except:
            pass
        ax.legend()
    
    def plotAllLongitudinalProfiles(self):
        while True:
            try:
                self.plotLongitudinalProfile()
                self.next()
            except:
                return
    
    def plotFullEvent(self):
        # the weird loc[evt:evt] is to get a one-row dataframe (otherwisxe we get a Series, which converts everything to float)
        return EventDisplay(self.peaks_df_subset.loc[self.cur_eventInternal:self.cur_eventInternal], self.eventLoader)

# class EventCountComputation(BaseComputation):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(neededBranches=["beamEnergy"], **kwargs)
#         self.countPerBeamEnergy = dict()
    

class CamelFinderDriver:
    def __init__(self, datatype, clueParams, version, histFolderBase='/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d') -> None:
        self.reader = ClueNtupleReader(version, clueParams, datatype)
        self.datatype = datatype
        self.eventLoader = EventLoader(self.reader.pathToFile)
    

    def findPeaks(self, peakFindindSettings:dict=dict(distance=5,  width=1.5, rel_height=0.7)):
        """ First step : find "peaks", which are actually dips, using scipy.signal.find_peaks. Code is in peak_finding.py 
        Fills self.peaks_df and self.energyPerLayer_df 
        """
        self.peakFindingSettings = peakFindindSettings
        peaksDf_comp = MakePeaksDfComputation(peakFindindSettings)
        computeAllFromTree(self.reader.tree, [peaksDf_comp], tqdm_options=dict(desc="Finding peaks"))

        self.peaks_df, self.energyPerLayer_df = peaksDf_comp.getResult()

        print(f"Peak finding efficiency : {len(self.peaks_df)/self.reader.tree.num_entries:.1%}")
    
    def addInfoToPeaks(self):
        """ Add useful columns to peaks dataframe """
        self.peaks_df = self.peaks_df.join(self.energyPerLayer_df.rechits_energy_sum_perLayer.groupby("eventInternal").max().rename("rechits_energy_maxLayer"))
    
    def applyFilters(self):
        """ Apply filter to found peaks. Currently : filter dips whose bordering peaks are very low, and filter peaks due to noisy cells 
        Fills self.peaks_df_filtered
        """
        filterApplier = FilterApplier(self.peaks_df, self.energyPerLayer_df, 
            [FilterLowBaseHeight(minFractionOfMaxLayerEnergy=0.3),
            FilterNoisyCellsPeaksUsingBases(), FilterDipsNotInCenter()])
        self.peaks_df_filtered = filterApplier.applyAllFilters()
    
    def extractMostProminentPeak(self):
        """ Extract from found peaks the one with the most prominence. 
        Fills self.peaks_df_mostProminent
        """
        self.peaks_df_mostProminent:pd.DataFrame = filtering.extractMostProminentPeak(self.peaks_df_filtered)
    

    def makeHistogram(self) -> hist.Hist:
        """ Make a 3D histogram self.h_lengthProminence with axes : 
         - beamEnergy
         - dipWidthQuantile : the width of the dip at the quantile (see self.peakFindingSettings) height between the bottom of the dip and the lowest of the two adjoining maximas
         - dipProminence : the prominence of the dip (ie height between minima and lowest adjoining maxima)
        Also makes self.h_length_prominenceNormalized, which replaces dipProminence by dipProminenceNormalized (the dip prominence normalized by the mean maximum layer energy at this beam energy)
        Returns a tuple of these two histograms
        """
        dipWidth_axis = hist.axis.Regular(20, 0, 10, name="dipWidthQuantile", label=f"Width of dip, at {self.peakFindingSettings['rel_height']:.0%} quantile height (in layers)")
        self.h_lengthProminence = hist.Hist(beamEnergiesAxis(), 
            dipWidth_axis,
            hist.axis.Regular(30, 0., 0.07, name="dipProminence", label="Dip prominence (GeV)"))
        self.h_length_prominenceNormalized = hist.Hist(beamEnergiesAxis(), dipWidth_axis,
            hist.axis.Regular(30, 0., 1.2, name="dipProminenceNormalized", label="Dip prominence, as fraction of mean maximum layer energy")
        )
        df = self.peaks_df_mostProminent
        self.h_lengthProminence.fill(df.beamEnergy, df.widths, df.prominences)
        self.h_length_prominenceNormalized.fill(df.beamEnergy, df.widths, df.prominences / df.beamEnergy.map(self.maxLayerEnergyPerBeamEnergy))

        return self.h_lengthProminence, self.h_length_prominenceNormalized
        
    @property
    def maxLayerEnergyPerBeamEnergy(self) -> dict[int, float]:
        """ For each beam energy, find the maximum among the mean energy sum per layer """
        energyPerLayer_h = (self.reader.histStore
            .get(HistogramId("RechitsEnergyReconstructedPerLayer", self.reader.clueParams, self.reader.datatype))
            .getHistogram(HistogramKind.PROFILE)
        )
        res = dict()
        for beamEnergy in energyPerLayer_h.axes["beamEnergy"]:
            res[beamEnergy] = np.max(energyPerLayer_h[{"beamEnergy":hist.loc(beamEnergy)}].view(flow=False).value)
        return res
    
    def makeAll(self):
        """ Do all the steps """
        self.findPeaks()
        self.addInfoToPeaks()
        self.applyFilters()
        self.extractMostProminentPeak()
        self.makeHistogram()

    @cached_property
    def eventsPerBeamEnergy(self) -> hist.Hist:
        return (self.reader.histStore
            .get(HistogramId("EventsPerBeamEnergy", self.reader.clueParams, self.reader.datatype))
            .getHistogram(HistogramKind.COUNT)
        )

    def plotHistogram(self, beamEnergy:int|list[int], ax=None, normalizeEventCount=False, normalizeProminence=True):
        """ Plot the histogram self.h_lengthProminence """
        if ax is None:
            fig, ax = plt.subplots()
        plt.style.use(hep.style.CMS)

        beamEnergy_orig = beamEnergy
        if not isinstance(beamEnergy, list):
            beamEnergies = [beamEnergy]
        else:
            beamEnergies = beamEnergy
        
        if normalizeProminence:
            h = self.h_length_prominenceNormalized
            prominenceAxisName = "dipProminenceNormalized"
        else:
            h = self.h_lengthProminence
            prominenceAxisName = "dipProminence"

        h = h[{"beamEnergy":[hist.loc(x) for x in beamEnergies], #"dipProminence":hist.rebin(2)
            }].project("dipWidthQuantile", prominenceAxisName)
        
        if normalizeEventCount:
            h = h / sum(self.eventsPerBeamEnergy[hist.loc(beamEnergy)] for beamEnergy in beamEnergies)

        artists = hep.hist2dplot(h, norm=matplotlib.colors.LogNorm(),
                    flow="none", ax=ax)
        
        if normalizeProminence:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            ax.set_ylim(0, 0.8)
            ax.set_ylabel("Dip prominence (norm. to max mean layer energy)")
        else:
            ax.set_ylim(0, 5/100)

        if normalizeEventCount:
            artists.cbar.ax.set_ylabel("Fraction of events")
            artists.cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        else:
            artists.cbar.ax.set_ylabel("Event count")
        
        ax.set_xlabel(f"Width of dip, at {self.peakFindingSettings['rel_height']:.0%} quantile height (in layers)")

        if self.datatype == "data":
            hep.cms.text("Preliminary", fontsize=20, ax=ax)
        else:
            hep.cms.text("Simulation Preliminary", fontsize=20, ax=ax)
        
        hep.cms.lumitext(f"{beamEnergy_orig} GeV - $e^+$ TB", fontsize=20, ax=ax)
    
    def getPassingEventList(self) -> EventList:
        return EventList(self.peaks_df_mostProminent, self.energyPerLayer_df, self.eventLoader)