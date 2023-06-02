import os

import uproot
import numpy as np
import pandas as pd
import hist
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

from event_visualizer.event_index import EventLoader
from event_visualizer.notebook_visualizer import EventDisplay
from ntupleReaders.computation import computeAllFromTree
from hists.custom_hists import beamEnergiesAxis

from longitudinalProfile.camel.peak_finding import MakePeaksDfComputation
from longitudinalProfile.camel.filtering import FilterLowBaseHeight, FilterNoisyCellsPeaks, FilterApplier, extractMostProminentPeak
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
    
    def next(self):
        self.cur_index_i += 1
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
    
    def plotLongitudinalProfile(self):
        event_data = self._getCurrentEvent()
        plotIndividualProfile_scipy(event_data["energyPerLayer"], *event_data["peaksInfo"])
        try:
            #hep.cms.label("Preliminary")
            hep.cms.lumitext(f'{event_data["beamEnergy"]:.0f} GeV - ntuple {event_data["ntupleNumber"]} - event {event_data["event"]}')
        except:
            pass
    
    def plotFullEvent(self):
        # the weird loc[evt:evt] is to get a one-row dataframe (otherwisxe we get a Series, which converts everything to float)
        return EventDisplay(self.peaks_df_subset.loc[self.cur_eventInternal:self.cur_eventInternal], self.eventLoader)

class CamelFinderDriver:
    def __init__(self, datatype, clueParams, version, histFolderBase='/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d') -> None:
        self.datatype = datatype
        clueClustersFile = os.path.join(histFolderBase, version, clueParams, datatype, "CLUE_clusters.root")
        self.tree = uproot.open(clueClustersFile + ":clusters")
        self.eventLoader = EventLoader(clueClustersFile)
    

    def findPeaks(self, peakFindindSettings:dict=dict(distance=5,  width=1.5)):
        """ First step : find "peaks", which are actually dips, using scipy.signal.find_peaks. Code is in peak_finding.py 
        Fills self.peaks_df and self.energyPerLayer_df 
        """
        peaksDf_comp = MakePeaksDfComputation(peakFindindSettings)
        computeAllFromTree(self.tree, [peaksDf_comp], tqdm_options=dict(desc="Finding peaks"))

        self.peaks_df, self.energyPerLayer_df = peaksDf_comp.getResult()

        print(f"Peak finding efficiency : {len(self.peaks_df)/self.tree.num_entries:.1%}")
    
    def addInfoToPeaks(self):
        """ Add useful columns to peaks dataframe """
        self.peaks_df = self.peaks_df.join(self.energyPerLayer_df.rechits_energy_sum_perLayer.groupby("eventInternal").max().rename("rechits_energy_maxLayer"))
    
    def applyFilters(self):
        """ Apply filter to found peaks. Currently : filter dips whose bordering peaks are very low, and filter peaks due to noisy cells 
        Fills self.peaks_df_filtered
        """
        filterApplier = FilterApplier(self.peaks_df, self.energyPerLayer_df, 
            [FilterLowBaseHeight(minFractionOfMaxLayerEnergy=0.5),
            FilterNoisyCellsPeaks()])
        self.peaks_df_filtered = filterApplier.applyAllFilters()
    
    def extractMostProminentPeak(self):
        """ Extract from found peaks the one with the most prominence. 
        Fills self.peaks_df_mostProminent
        """
        self.peaks_df_mostProminent = extractMostProminentPeak(self.peaks_df_filtered)
    

    def makeHistogram(self) -> hist.Hist:
        """ Make a 3D histogram with axes : 
         - beamEnergy
         - dipLengthMedian : the width of the dip at the median height between the bottom of the dip and the lowest of the two adjoining maximas
         - dipRelativeProminence : the prominence of the dip (ie height between minima and lowest adjoining maxima), as fraction of beam energy
        Fillfs self.h_lengthProminence (also returns it)
        """
        self.h_lengthProminence = hist.Hist(beamEnergiesAxis(), 
            hist.axis.Regular(20, 0, 8, name="dipLengthMedian", label="Length of dip, at median height (in layers)"), 
            hist.axis.Regular(30, 0., 0.07, name="dipRelativeProminence", label="Dip prominence as fraction of beam energy"))
        df = self.peaks_df_mostProminent
        self.h_lengthProminence.fill(df.beamEnergy, df.widths, df.prominences/df.beamEnergy)

        return self.h_lengthProminence
    
    def makeAll(self):
        """ Do all the steps """
        self.findPeaks()
        self.addInfoToPeaks()
        self.applyFilters()
        self.extractMostProminentPeak()
        self.makeHistogram()


    def plotHistogram(self, beamEnergy:int|list[int]):
        """ Plot the histogram self.h_lengthProminence """
        plt.figure()
        beamEnergy_orig = beamEnergy
        if not isinstance(beamEnergy, list):
            beamEnergy = [beamEnergy]
        h = self.h_lengthProminence
        h = h[{"beamEnergy":[hist.loc(x) for x in beamEnergy], #"dipProminence":hist.rebin(2)
            }].project("dipLengthMedian", "dipRelativeProminence")
        hep.hist2dplot(h, #norm=matplotlib.colors.LogNorm()
                    flow="none")
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        if self.datatype == "data":
            hep.cms.text("Preliminary")
        else:
            hep.cms.text("Simulation Preliminary")
        
        hep.cms.lumitext(f"{beamEnergy_orig} GeV - $e^+$ TB")
    
    def getPassingEventList(self) -> EventList:
        return EventList(self.peaks_df_mostProminent, self.energyPerLayer_df, self.eventLoader)