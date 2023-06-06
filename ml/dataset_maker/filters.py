import typing
import functools
import os

import uproot
import awkward as ak
import numpy as np
import pandas as pd

from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from ntupleReaders.computation import BaseComputation, computeAllFromTree, ComputationToolBase, Report
from ntupleReaders.tools import DataframeComputationsToolMaker
from hists.dataframe import DataframeComputations



class FilteringComputation(BaseComputation):
    """ Abstract class for event filtering operations. 
    Child classes need to implement either 
     - computeFilterArray_comp(self, comp:DataframeComputations)
     - or def computeFilterArray_awkward(self, array:ak.Array)
    these function can either return a numpy bool array, or a pd.Series of bool (in this case the series will be reindexed, filling with True)
    """

    def __init__(self, neededBranches:list[str]) -> None:
        super().__init__(neededBranches=neededBranches, neededComputationTools=[DataframeComputationsToolMaker(rechits_columns=["rechits_energy", "rechits_layer"])])
        self.filterArrays:list[np.array[bool]] = []
        """ list of booleans arrays, to be concatenated to form one array
        Each element maps by index to an event in the input tree
        True means the event passes the filter, False it does not
        """


    def processBatch(self, array:ak.Array, report:Report, computationTools:dict[typing.Type[ComputationToolBase], ComputationToolBase]) -> None:
        comp:DataframeComputations = computationTools[DataframeComputations]
        try:
            filterArray = self.computeFilterArray_comp(comp)
        except AttributeError:
            filterArray = self.computeFilterArray_awkward(array)

        if isinstance(filterArray, pd.Series):
            # in case some rows got lost in joins, we reindex the series
            # filling with True
            filterArray = filterArray.reindex(index=pd.RangeIndex(stop=len(array)), fill_value=True).to_numpy()
        self.filterArrays.append(filterArray)

    #def computeFilterArray_comp(self, comp:DataframeComputations) -> pd.Series[bool]:
    #    pass

    #def computeFilterArray_awkward(self, array:ak.Array) -> np.array[bool]:
    #    pass

    def getResult(self) -> np.ndarray[bool]:
        """ Compute the full filter array. Same length as the input tree. """
        return np.concatenate(self.filterArrays)
    
    def getEfficiency(self):
        """ Compute efficiency of filter (fraction of events passing the filter) """
        return sum((np.count_nonzero(a) for a in self.filterArrays))/sum((len(a) for a in self.filterArrays))

class LowRecoEnergyFilter(FilteringComputation):
    """ Filter retaining only events having at least a fraction of the synchrotron-corrected beam energy """

    def __init__(self, minFractionOfSynchrotonBeamEnergy=0.8) -> None:
        super().__init__(neededBranches=["rechits_energy", "beamEnergy"])
        self.minFraction = minFractionOfSynchrotonBeamEnergy
    
    def computeFilterArray_comp(self, comp:DataframeComputations) -> ak.Array[bool]:
        df = comp.rechits_totalReconstructedEnergyPerEvent
        df = df.assign(synchrotronBeamEnergy=comp.beamEnergy.synchrotronBeamEnergy)
        return df.rechits_energy_sum >= self.minFraction * df.synchrotronBeamEnergy
    
class LowMainTracksterEnergyFilter(FilteringComputation):
    """ Filter retaining only events where the main CLUE3D trackster has at least a fraction of the synchrotron-corrected beam energy """

    def __init__(self, minFractionOfSynchrotonBeamEnergy=0.7) -> None:
        super().__init__(neededBranches=["beamEnergy", "clus3D_x", "clus3D_y", "clus3D_z", "clus3D_energy", "clus3D_size", "beamEnergy"])
        self.minFraction = minFractionOfSynchrotonBeamEnergy
    
    def computeFilterArray_comp(self, comp:DataframeComputations) -> ak.Array[bool]:
        df = comp.clusters3D_largestCluster.reset_index("clus3D_id", drop=True)
        df = df.join(comp.beamEnergy.synchrotronBeamEnergy)
        return df.clus3D_energy >= self.minFraction * df.synchrotronBeamEnergy


class FilterEmptyEvents(FilteringComputation):
    """ Filter dropping empty events (happen only in simulation, quite rare) """

    def __init__(self) -> None:
        super().__init__(neededBranches=["rechits_energy"])

    def computeFilterArray_awkward(self, array:ak.Array) -> np.ndarray[bool]:
        return ak.num(array.rechits_energy, axis=1)>0


def makeDefaultFilters() -> list[FilteringComputation]:
    return [LowRecoEnergyFilter(), LowMainTracksterEnergyFilter(), FilterEmptyEvents()]


def applyAllFilters(tree:uproot.TTree, filters:list[FilteringComputation]=None):
    if filters is None:
        filters = makeDefaultFilters()
    
    computeAllFromTree(tree, filters, tqdm_options=dict(desc="Computing filters"))
    for filter in filters:
        print(filter.__class__.__name__ + f" - Efficiency {filter.getEfficiency():%}")
    
    return functools.reduce(lambda x, y : x&y, [filter.getResult() for filter in filters])
    
def makeAndSaveFilters(reader:ClueNtupleReader, fileName="default_filter"):
    filters = applyAllFilters(reader.tree)
    np.save(os.path.join(reader.pathToFolder, fileName), filters) # will append .npy to the filename