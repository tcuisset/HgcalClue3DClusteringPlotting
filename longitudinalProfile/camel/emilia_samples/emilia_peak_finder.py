import os
import glob
from functools import cached_property

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import hist

from longitudinalProfile.camel.peak_finding import *
from longitudinalProfile.camel.driver import CamelFinderDriver

emiliaBeamEnergy = 50

def readEmiliaSample(samplePath:str) -> pd.DataFrame:
    df = pd.read_csv(samplePath,
        names=["eventInternal", "rechits_id", "rechits_energy_raw", "rechits_x", "rechits_y", "rechits_z"], header=0,
        index_col=[0, 1]
    )
    return df

def addLayerAndEnergyCorrectionToSample(df):
    return (df.assign(
            rechits_layer=df.rechits_z.map(zToLayerMapping),
            rechits_energy=df.rechits_energy_raw / 645 * emiliaBeamEnergy, # dE/dx correction to bring reco energy at 50 GeV
            beamEnergy=emiliaBeamEnergy
        )
        .reset_index("rechits_id")
        .set_index("rechits_layer", append=True)
    )

class EmiliaSampleReader:
    def __init__(self, folderName="elm_E50_theta0") -> None:
        self.folderName = folderName
        self.pathToFolder = os.path.join("/data_CMS_upgrade/hgcnn/d1", folderName)
        self.loadedSamples:dict[int, pd.DataFrame] = dict()

    @cached_property
    def getHitsFiles(self) -> list[str]:
        return glob.glob(os.path.join(self.pathToFolder, "sensors_d1*.dat"))
    
    @property
    def sampleCount(self) -> int:
        return len(self.getHitsFiles)

    def getSample(self, i:int) -> pd.DataFrame:
        try:
            return self.loadedSamples[i]
        except KeyError:
            sample = addLayerAndEnergyCorrectionToSample(readEmiliaSample(self.getHitsFiles[i]))
            self.loadedSamples[i] = sample
            return sample
    
    def unloadSamples(self):
        del self.loadedSamples
        self.loadedSamples = dict()

zToLayerMapping = dict(zip([1005.76, 1014.68, 1023.6 , 1032.52, 1041.44, 1050.36, 1059.28,
       1068.2 , 1077.12, 1086.04, 1094.96, 1103.88, 1112.8 , 1121.72,
       1130.64, 1139.56, 1148.48, 1157.4 , 1166.32, 1175.24, 1184.16,
       1193.08, 1202.  , 1210.92, 1219.84], range(1, 25+1)))



def processEmiliaSample(df:pd.DataFrame) -> pd.DataFrame:
    ##### Sum of all rechits energy per event and per layer, but with all layers present (filled with zeroes if necessary)
    energySumPerLayer = df.rechits_energy.groupby(by=["eventInternal", "rechits_layer"]).sum()

    # We build the cartesian product event * layer
    newIndex = pd.MultiIndex.from_product([energySumPerLayer.index.levels[0], energySumPerLayer.index.levels[1]])

    # Reindex the dataframe, this will create new rows as needed, filled with zeros
    # Make sure only columns where 0 makes sense are included (not beamEnergy !)
    energySumPerLayer = energySumPerLayer.reindex(newIndex, fill_value=0).rename("rechits_energy_sum_perLayer")
    
    return_df = energySumPerLayer.to_frame().join(energySumPerLayer.groupby("eventInternal").sum().rename("rechits_energy_sum"))
    #return 
    # return (df.assign(rechits_energy_sum_perLayer=energySumPerLayer)
    #     .join(df.rechits_energy.groupby("eventInternal").sum().rename("rechits_energy_sum"))
    # )


    ############ ratio of first to second most energetic hit
    df_ratioFirstToSecond = (df[["rechits_energy"]]
        # Select the first two rechits in energy per layer
        .sort_values(["eventInternal", "rechits_layer", "rechits_energy"], ascending=[True, True, False])
        .groupby(["eventInternal", "rechits_layer"]).head(2)
    )
    # unstack the dataframe
    df_firstAndSecond = (df_ratioFirstToSecond
        # Make a cumulative count, which is 0 for the most energetic hit, 1 for the second (for each event and layer)
        .assign(cumcount=df_ratioFirstToSecond.groupby(["eventInternal", "rechits_layer"]).cumcount())
        .set_index("cumcount", append=True)
        # Unstack the dataframe (filling rechits_id with -1 in case of only one hit)
        .unstack(fill_value=-1)
    )
    ratio = df_firstAndSecond.rechits_energy[0]/df_firstAndSecond.rechits_energy[1]
    ratio = ratio[ratio > 0].rename("rechits_ratioFirstToSecondMostEnergeticHitsPerLayer") # drop cases where there is only one rechit on layer (then rechits_energy[1] is -1)

    return return_df.join(ratio).assign(beamEnergy=emiliaBeamEnergy)


class EmiliaPeakFinder:
    def __init__(self, sampleReader:EmiliaSampleReader, settings, peakFindingFunction=find_peaks_reverse) -> None:
        """ Parameters :
         - settings : passed to scipy.signal.find_peaks
         - peakFindingFunction : function to find peaks, should be find_peaks_reverse
        """
        self.sampleReader = sampleReader
        self.dfList = []
        self.perLayerDfList = []
        self.settings = settings
        self.peakFindingFunction = peakFindingFunction
        self.num_entries = 0
        """ Number of events read """

    def _loadSample(self, sample_i:int):
        sample_df = self.sampleReader.getSample(sample_i)
        self.num_entries += len(np.unique(sample_df.index.get_level_values(0)))
        energyPerLayer_df = processEmiliaSample(sample_df)
        peaks_series = self.peakFindingFunction(energyPerLayer_df, self.settings)
        # We make two separate dataframes. One indexed solely by eventInternal : 
        self.dfList.append(peaks_series.to_frame())
        
        # And another indexed by eventInternal, rechits_layer :
        # (Keeping only events in energyPerLayer_df that got saved in peaks_series)
        energyPerLayer_df = energyPerLayer_df.reset_index("rechits_layer")
        self.perLayerDfList.append(energyPerLayer_df[energyPerLayer_df.index.isin(peaks_series.index)])
    
    def loadAllSamples(self, sampleRange=None, tqdm_kwargs=dict()):
        if sampleRange is None:
            sampleRange = range(0, len(self.sampleReader.getHitsFiles))
        for sample_i in tqdm(sampleRange, **tqdm_kwargs):
            self._loadSample(sample_i)
            #self.sampleReader.loadedSamples.pop(sample_i-1, None)
    
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


class EmiliaCamelFinderDriver(CamelFinderDriver):
    def __init__(self, reader:EmiliaSampleReader) -> None:
        self.reader = reader
        self.datatype = "simulation_emilia"
        self.eventLoader = None
        self.simulation = True
    
    def findPeaks(self, peakFindindSettings:dict=dict(distance=5,  width=1.5, rel_height=0.7), sampleRange=None, tqdm_kwargs=dict()):
        """ First step : find "peaks", which are actually dips, using scipy.signal.find_peaks. Code is in peak_finding.py 
        Fills self.peaks_df and self.energyPerLayer_df 
        """
        self.peakFindingSettings = peakFindindSettings
        peaks_comp = EmiliaPeakFinder(self.reader, peakFindindSettings)
        peaks_comp.loadAllSamples(sampleRange=sampleRange, tqdm_kwargs=tqdm_kwargs)
        self.peaks_df, self.energyPerLayer_df = peaks_comp.getResult()

        print(f"Peak finding efficiency : {len(self.peaks_df)/peaks_comp.num_entries:.1%}")

        self.eventsPerBeamEnergy = hist.Hist(hist.axis.IntCategory([emiliaBeamEnergy], name="beamEnergy"))
        self.eventsPerBeamEnergy.view()[0] = peaks_comp.num_entries

    def extractMostProminentPeak(self):
        super().extractMostProminentPeak()
        self.peaks_df_mostProminent["beamEnergy"] = emiliaBeamEnergy

    @property
    def maxLayerEnergyPerBeamEnergy(self) -> dict[int, float]:
        """ For each beam energy, find the maximum among the mean energy sum per layer """
        return {50 : 5.087} # hardcoded for now, see in longitudinalProfile/camel/emilia_samples/max_mean_layer_energy.ipynb for the notebook used to compute the value 
    