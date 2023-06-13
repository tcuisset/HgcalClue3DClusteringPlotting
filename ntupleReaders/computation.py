import typing
from collections import defaultdict

import numpy as np
import uproot
import uproot.behaviors.TBranch
Report = uproot.behaviors.TBranch.Report
import awkward as ak
from tqdm.auto import tqdm

from hists.dataframe import DataframeComputations

class ComputationToolBase:
    pass


class ComputationToolMakerBase:
    def __init__(self, neededBranches=[]) -> None:
        self.neededBranches = neededBranches
        """ Branches needed by the computation """

    def instantiateComputationTool(self, array:ak.Array, report:Report) -> ComputationToolBase:
        """ Create a ComputationTool instance, called once per batch and per filter 
        Note that there may be less events in array than report suggests, since events could have been filtered
        """
        pass

class EventFilterBase:
    """ Base class for a filter. Must implement applyFilter """
    def __init__(self, neededBranches=[]) -> None:
        self.neededBranches = neededBranches
        """ List of branches needed for the the filter (usually left empty) """

    
    def applyFilter(self, array:ak.Array, report:Report) -> ak.Array:
        """ Applies the filter. Should return a subset of rows of array """
        raise NotImplementedError()

class NoFilter(EventFilterBase):
    """ Filter that does not filter anything """
    def applyFilter(self, array:ak.Array, report:Report) -> ak.Array:
        """ Applies the filter. Should return a subset of rows of array """
        return array

class NumpyArrayFilter(EventFilterBase):
    """ Filter that uses a numpy boolean array to filter based on indices """
    def __init__(self, filterArray:np.ndarray[bool]) -> None:
        super().__init__()
        self.filterArray = filterArray

    def applyFilter(self, array: ak.Array, report:Report) -> ak.Array:
        filterArray_currentIteration = self.filterArray[report.global_entry_start:report.global_entry_stop]
        return array[filterArray_currentIteration]

class BeamEnergyFilter(EventFilterBase):
    """ Filter to select only beam energies given in argument """
    def __init__(self, beamEnergiesToSelect:list[int]) -> None:
        super().__init__(neededBranches=["beamEnergy"])
        self.beamEnergiesToSelect = beamEnergiesToSelect

    def applyFilter(self, array: ak.Array, report:Report) -> ak.Array:
        return array[np.isin(array.beamEnergy, self.beamEnergiesToSelect)]

class ChainedFilter(EventFilterBase):
    """ Filter that applies a series of filters one by one (events have to pass all filters) 
    Warning : only the first filter can rely on absolute indices
    """
    def __init__(self, filterList:list[EventFilterBase]) -> None:
        super().__init__()
        self.filterList = filterList

    def applyFilter(self, array: ak.Array, report:Report) -> ak.Array:
        # give the report only to the first filter
        array = self.filterList[0].applyFilter(array, report)
        for filterObject in self.filterList[1:]:
            array = filterObject.applyFilter(array, None)
        return array

class BaseComputation:
    def __init__(self, neededBranches=[], eventFilter = NoFilter(), neededComputationTools:list[ComputationToolMakerBase]=[]) -> None:
        self.neededBranches:list[str] = neededBranches
        """ List of branches needed for the computation """

        self.eventFilter = eventFilter

        self.neededComputationTools:list[ComputationToolMakerBase] = neededComputationTools
        """ List of computation tools makers (shared among computations that have the same filter) needed for this computation """

    def processBatch(self, array:ak.Array, report:Report, computationTools:dict[typing.Type[ComputationToolBase], ComputationToolBase]) -> None:
        pass


def computeAllFromTree(tree:uproot.TTree|list[uproot.TTree], computations:list[BaseComputation], tqdm_options:dict=dict()):
    """ Process all given Computation objects from all events in tree 
    Parameters : 
     - tree : an uproot TTree to process, or a list of them
     - computations : a list of computations (they must satifsfy the BaseComputation model)
     - tqdm_options : dict of keyword args passed to tqdm. You can pass for example desc, or count (in case you provide an generator for tree which has no __len__)
     """
    neededBranches = set()
    computationsPerFilter:dict[EventFilterBase, list[BaseComputation]] = defaultdict(list)
    computationToolMakers:dict[EventFilterBase, set[ComputationToolMakerBase]] = defaultdict(set)
    for comp in computations:
        neededBranches.update(comp.neededBranches)
        neededBranches.update(comp.eventFilter.neededBranches)

        computationsPerFilter[comp.eventFilter].append(comp)
        computationToolMakers[comp.eventFilter].update(comp.neededComputationTools)

        for compToolMaker in comp.neededComputationTools:
            try:
                neededBranches.update(compToolMaker.neededBranches)
            except TypeError:
                pass # in case compToolMaker.neededBranches is None
    

    if len(neededBranches) == 0:
        raise ValueError("needBranches was empty : aborting (you should set needBranches in Computation classes)")

    def processArray(array:ak.Array, report:Report):
        assert len(array) == report.global_entry_stop-report.global_entry_start, "Mismatching lengths in tree"

        for eventFilter in computationsPerFilter.keys():
            filteredArray = eventFilter.applyFilter(array, report)

            # instantiate tools for current filter
            computationTools = {compToolMaker : compToolMaker.instantiateComputationTool(filteredArray, report) for compToolMaker in computationToolMakers[eventFilter]}
            
            for comp in computationsPerFilter[eventFilter]:
                # making a dict Type[ComputationToolBase] -> ComputationToolBase
                compToolsForCurrentComp = {computationTools[compToolMaker].__class__ : computationTools[compToolMaker] for compToolMaker in comp.neededComputationTools}
                try:
                    comp.processBatch(array=filteredArray, report=report, computationTools=compToolsForCurrentComp)
                except TypeError:
                    try:
                        comp.processBatch(filteredArray, compToolsForCurrentComp)
                    except TypeError:
                        comp.processBatch(filteredArray)

    if isinstance(tree, uproot.TTree):
        with tqdm(total=tree.num_entries, **tqdm_options) as pbar:
            for (array, report) in tree.iterate(step_size="200MB", library="ak", report=True, filter_name=list(neededBranches)):
                processArray(array, report)
                pbar.update(report.stop-report.start)
    else:
        for array, report in tqdm(uproot.iterate(tree, step_size="200MB", library="ak", report=True, filter_name=list(neededBranches)),
                            **tqdm_options):
            processArray(array, report)


