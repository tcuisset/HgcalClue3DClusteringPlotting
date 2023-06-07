from operator import itemgetter

import numpy as np
import pandas as pd
import scipy


class RowObject:
    def __init__(self, df_energyPerLayer:pd.DataFrame, row_tuple) -> None:
        self.df_energyPerLayer = df_energyPerLayer
        self.row_tuple = row_tuple
    
    def __getattr__(self, name):
        return getattr(self.row_tuple, name)

    def getEnergyAtLayer(self, layer) -> float:
        return self.df_energyPerLayer.at[(self.eventInternal, layer), "rechits_energy_sum_perLayer"]

    def getEnergyForAllLayers(self) -> pd.Series:
        return self.df_energyPerLayer.loc[(self.eventInternal, slice(None)), "rechits_energy_sum_perLayer"]

class FilterBase:
    def _applyFct(self, row:RowObject) -> object:
        """ Function passed to dataframe.apply 
        Parameters : - row : an object holding
         - row.peaks_info
         - row.rechits_energy_maxLayer : energy of the layer with max energy
         - row.getEnergyAtLayer(layer) : fct getting the energy at layer
        Returns : 
         - None to drop the event
         - anything else : will be taken as an updated (skimmed) peaks_info
        """
        pass


class FilterLowBaseHeight(FilterBase):
    """ Remove peaks where the energy of the lowest peak is less than fraction times the maximum layer energy """
    def __init__(self, minFractionOfMaxLayerEnergy) -> None:
        self.fraction = minFractionOfMaxLayerEnergy
    
    def _applyFct(self, row:RowObject) -> object:
        peaks, properties = row.peaks_info
        peaks_indices_selection = []
        for peak, left_base, right_base in zip(peaks, properties["left_bases"], properties["right_bases"]):
            # map 0-based to 1-based indexing
            energyAtLowestBase = min(row.getEnergyAtLayer(left_base+1), row.getEnergyAtLayer(right_base+1))
            peaks_indices_selection.append(energyAtLowestBase > self.fraction * row.rechits_energy_maxLayer)
        
        if not np.any(peaks_indices_selection):
            return None # in case no peaks pass selections
        
        # select from input peaks_info only the peaks whose indices are in peaks_indices_selection
        getter = itemgetter(np.nonzero(peaks_indices_selection)[0])
        return getter(peaks), {key : getter(val) for key, val in properties.items()}


class FilterNoisyCellsPeaks(FilterBase):
    """ Filter out events which have high and narrow peaks compatible with a noisy cell
    
    """
    def __init__(self, maxPeakWidth=2, quantileForMaxProminence=0.7, minRatioFirstToSecondRechit=5) -> None:
        self.maxPeakWidth = maxPeakWidth
        self.quantileForMaxProminence = quantileForMaxProminence
        self.minRatioFirstToSecondRechit = minRatioFirstToSecondRechit

    def _applyFct(self, row: RowObject) -> object:
        energyPerLayer = row.getEnergyForAllLayers()

        peaks, properties = scipy.signal.find_peaks(energyPerLayer, width=(0, self.maxPeakWidth), prominence=energyPerLayer.quantile(self.quantileForMaxProminence))
        
        if len(peaks) > 0:
            index = np.argmax(properties["prominences"]) # most prominent peak
            layer = peaks[index] + 1 # layer of most prominent peak
            ratioFirstToSecond = row.df_energyPerLayer.at[(row.eventInternal, layer), "rechits_ratioFirstToSecondMostEnergeticHitsPerLayer"]
            if ratioFirstToSecond > self.minRatioFirstToSecondRechit:
                return None
        
        return row.peaks_info

class FilterNoisyCellsPeaksUsingBases(FilterBase):
    """ Filter out events which have high and narrow peaks compatible with a noisy cell
    Does this by looking, for each dip, at the right and left bases layers. If on one of these layers,
    the ratio of first to second rechits energies (sorted by decreasing energies) is greater than a threshold, the dip is discarded
    """
    def __init__(self, minRatioFirstToSecondRechit=10) -> None:
        self.minRatioFirstToSecondRechit = minRatioFirstToSecondRechit

    def _applyFct(self, row: RowObject) -> object:
        peaks, properties = row.peaks_info

        peaks_indices_selection = []
        for peak, left_base, right_base in zip(peaks, properties["left_bases"], properties["right_bases"]):
            drop_peak = False
            # map 0-based to 1-based indexing
            for base_layer in (left_base+1, right_base+1):
                ratioFirstToSecond = row.df_energyPerLayer.at[(row.eventInternal, base_layer), "rechits_ratioFirstToSecondMostEnergeticHitsPerLayer"]
                if ratioFirstToSecond > self.minRatioFirstToSecondRechit:
                    drop_peak = True
            
            peaks_indices_selection.append(not drop_peak)

        if not np.any(peaks_indices_selection):
            return None # in case no peaks pass selections
        
        # select from input peaks_info only the peaks whose indices are in peaks_indices_selection
        getter = itemgetter(np.nonzero(peaks_indices_selection)[0])
        return getter(peaks), {key : getter(val) for key, val in properties.items()}

class FilterDipsNotInCenter(FilterBase):
    """ Filter out dips that are either at the very beginning or the very end of the shower 
    (these are mainly caused by abnormally high energetic cells at the first or last layer)
    """
    def __init__(self, minLayerPosition=3, maxLayerPosition=26) -> None:
        self.minLayer = minLayerPosition
        self.maxLayer = maxLayerPosition
    
    def _applyFct(self, row: RowObject) -> object:
        peaks, properties = row.peaks_info

        peaks_indices_selection = []
        for peak in peaks:
            # map 0-based to 1-based indexing
            peaks_indices_selection.append((peak+1 >= self.minLayer) and (peak+1 <= self.maxLayer))

        if not np.any(peaks_indices_selection):
            return None # in case no peaks pass selections
        
        # select from input peaks_info only the peaks whose indices are in peaks_indices_selection
        getter = itemgetter(np.nonzero(peaks_indices_selection)[0])
        return getter(peaks), {key : getter(val) for key, val in properties.items()}

class FilterIsolatedPeaks(FilterBase): # not used anymore
    """ Filter out dips which are bordered by an abnormal peak (for example a noisy cell) 
    In case : - the dip min is one layer out from one of the maximas
    - the energy in the dip min is greater than than the layer after the adjacent maxima
    then we drop the dip
    """
    def _applyFct(self, row:RowObject) -> object:
        peaks, properties = row.peaks_info
        def getEnergyAtLayer_wrap(layer):
            """ Gets the energy on layer if the layer exists, otherwise 0 """
            try:
                return row.getEnergyAtLayer(layer)
            except Exception:
                return 0
            
        indices = []
        for peak, left_base, right_base in zip(peaks, properties["left_bases"], properties["right_bases"]):
            if peak == right_base - 1 and getEnergyAtLayer_wrap((peak+1)) > getEnergyAtLayer_wrap((peak+1)+2):
                indices.append(False)
            elif peak == left_base + 1 and getEnergyAtLayer_wrap((peak+1)) > getEnergyAtLayer_wrap((peak+1)-2):
                indices.append(False)
            else:
                indices.append(True)
        
        if not np.any(indices):
            return None # in case no peaks pass selections

        # select from input peaks_info only the peaks whose indices are in peaks_indices_selection
        getter = itemgetter(np.nonzero(indices)[0])
        return getter(peaks), {key : getter(val) for key, val in properties.items()}



def applyFilter(filter, df_peaks:pd.DataFrame, df_energyPerLayer):
    out_series = df_peaks.peaks_info.copy()
    for row_i, row in enumerate(df_peaks.itertuples()):
        row_obj = RowObject(df_energyPerLayer, row)
        row_obj.eventInternal = row.Index
        out_series.iat[row_i] = filter(row_obj)

    return out_series

class FilterApplier:
    """ Main class that manages the application of filters """
    def __init__(self, df_peaks:pd.DataFrame, df_energyPerLayer:pd.DataFrame, filters:list[FilterBase]) -> None:
        self.df_peaks = df_peaks
        self.df_energyPerLayer = df_energyPerLayer
        self.cur_df_peaks = df_peaks
        self.filters = filters
    
    def applySingleFilter(self, filter:FilterBase|int) -> tuple[pd.Series, float]:
        """ Apply a single filter
        Returns : efficiency"""
        if isinstance(filter, int):
            filter = self.filters[filter]
        peaks_info_series = applyFilter(filter._applyFct, self.cur_df_peaks, self.df_energyPerLayer)
        dropped_count = peaks_info_series.isna().sum()
        efficiency = (len(peaks_info_series) - dropped_count)/len(peaks_info_series)

        if dropped_count > 0:
            # re-join with peaks_df to update information and drop filtered-out rows
            new_df_peaks = self.cur_df_peaks.drop(columns="peaks_info").join(peaks_info_series.dropna(), how="right")
        else:
            # we can just braodcast here
            new_df_peaks = self.cur_df_peaks.drop(columns="peaks_info").assign(peaks_info=peaks_info_series)
        self.cur_df_peaks = new_df_peaks
        return efficiency
    
    def getPassingFailingFilter(self, filter:FilterBase|int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Returns a tuple with (passing, failing) events """
        if isinstance(filter, int):
            filter = self.filters[filter]
        peaks_info_series = applyFilter(filter._applyFct, self.cur_df_peaks, self.df_energyPerLayer)
        return self.cur_df_peaks[~peaks_info_series.isna()], self.cur_df_peaks[peaks_info_series.isna()]

    def applyAllFilters(self):
        """ Apply given filters consecutively, showing efficiency at each step """
        self.cur_df_peaks = self.df_peaks
        for filter in self.filters:
            efficiency = self.applySingleFilter(filter)
            print(filter.__class__.__name__ + f" - efficiency {efficiency:.1%}")
        
        return self.cur_df_peaks


def extractMostProminentPeak(peaks_df:pd.DataFrame) -> pd.DataFrame:
    """ Select the peak with the highest prominence, and extract its properties into dataframe columns """
    def applyFct(peaks_info):
        peaks, properties = peaks_info
        index = np.argmax(properties["prominences"])
        return pd.Series([peaks[index]] + [l[index] for l in properties.values()], index=["peak_index"]+list(properties.keys()))
    
    return peaks_df.drop(columns="peaks_info").join(peaks_df.peaks_info.apply(applyFct), how="right")
