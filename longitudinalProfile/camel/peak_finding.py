import scipy
import pandas as pd
import awkward as ak

from hists.dataframe import DataframeComputations
from ntupleReaders.computation import BaseComputation, computeAllFromTree

def computeEnergyPerLayerDf(comp:DataframeComputations):
    """ Make dataframe with :
    Index : eventInternal, rechits_layer
    Columns : 
     - rechits_energy_sumPerLayer : sum of all rechits energies on each event and layer
     - rechits_energy_sum : sum of all rechits energies per event
     - rechits_ratioFirstToSecondMostEnergeticHitsPerLayer : ratio of first to second most energetic hit per layer and event
    """
    energySumPerLayer_df = (comp
        .rechits_totalReconstructedEnergyPerEventLayer_allLayers(joinWithBeamEnergy=False)
        .set_index("rechits_layer", append=True)
    )
    energySumPerLayer_df = energySumPerLayer_df.join(energySumPerLayer_df
                .groupby(by="eventInternal")
                .agg(
                    rechits_energy_sum=pd.NamedAgg(column="rechits_energy_sum_perLayer", aggfunc="sum"),
                ))
    return energySumPerLayer_df.join(comp.rechits_ratioFirstToSecondMostEnergeticHitsPerLayer)

def find_peaks(df:pd.DataFrame, settings:dict):
    """ Find peaks in longitudinal profile. Selects events with two peaks. (NB: this function is not used anymore, use find_peaks_reverse)"""
    def apply_fct(grouped_series:pd.Series):
        peaks, properties = scipy.signal.find_peaks(grouped_series, **settings)
        if len(peaks) < 2:
            return None
        else:
            return peaks, properties
    return df.rechits_energy_sum_perLayer.groupby("eventInternal").apply(apply_fct).dropna().rename("peaks_info")

def find_peaks_reverse(df:pd.DataFrame, settings:dict):
    """ Find a dip in the longitudinal profile, by flipping the profile upside down and finding a peak there.
    Parameters : 
     - df : dataframe from computeEnergyPerLayerDf
     - settings : settings passed to scipy.signal.find_peaks
    Returns : a pandas Series indexed by eventInternal, with the result of scipy.signal.find_peaks. Only events with peaks are kept
    """
    def apply_fct(grouped_series:pd.Series):
        """ Function applied to a series of all rechits energies (indexed by layer) in a given event """
        peaks, properties = scipy.signal.find_peaks(-grouped_series, **settings) # note the minus sign
        if len(peaks) == 0:
            return None # no peak found
        else:
            return peaks, properties
    return (df.rechits_energy_sum_perLayer
            .groupby("eventInternal")
            .apply(apply_fct)
            .dropna() # remove events with no peak found
            .rename("peaks_info")
    )


class MakePeaksDfComputation(BaseComputation):
    """ Makes a dataframe holding all camel shower candidates """
    neededBranches = ["beamEnergy", "event", "ntupleNumber", "rechits_energy", "rechits_layer"]
    def __init__(self, settings, peakFindingFunction=find_peaks_reverse) -> None:
        """ Parameters :
         - settings : passed to scipy.signal.find_peaks
         - peakFindingFunction : function to find peaks, should be find_peaks_reverse
        """
        self.dfList = []
        self.perLayerDfList = []
        self.settings = settings
        self.peakFindingFunction = peakFindingFunction

    def process(self, array: ak.Array) -> None:
        comp = DataframeComputations(array, rechits_columns=["rechits_energy", "rechits_layer"])
        energyPerLayer_df = computeEnergyPerLayerDf(comp)
        peaks_series = self.peakFindingFunction(energyPerLayer_df, self.settings)
        
        # We make two separate dataframes. One indexed solely by eventInternal : 
        self.dfList.append(pd.merge(peaks_series, comp.ntupleEvent, how="left", left_index=True, right_index=True))
        
        # And another indexed by eventInternal, rechits_layer :
        # (Keeping only events in energyPerLayer_df that got saved in peaks_series)
        energyPerLayer_df = energyPerLayer_df.reset_index("rechits_layer")
        self.perLayerDfList.append(energyPerLayer_df[energyPerLayer_df.index.isin(peaks_series.index)])
    
    def getResult(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Gets a pair of dataframe, the first is info on peaks (indexed by eventInternal)
        The second is the energy per layer in all events selected in the first df
        """
        # Use concat with keys= so we can differentiate event nb from different batches
        keys = list(range(len(self.dfList)))
        # the reset_index will create a unique index over all batches
        concatDf = pd.concat(self.dfList, keys=keys).reset_index(names=["batchNumber", "eventInBatch"]).rename_axis(index="eventInternal")
        perLayer_concat = pd.concat(self.perLayerDfList, keys=keys)

        # Now join to the per layer df so we map [batchNumber, eventInBatch] to the unique index in concat_df
        perLayer_indexed = concatDf[["batchNumber", "eventInBatch"]].join(perLayer_concat, on=["batchNumber", "eventInBatch"]).set_index("rechits_layer", append=True)
        return concatDf.drop(columns=["batchNumber", "eventInBatch"]), perLayer_indexed.drop(columns=["batchNumber", "eventInBatch"])