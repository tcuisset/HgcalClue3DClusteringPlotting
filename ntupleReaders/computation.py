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
        """ Create a ComputationTool instance, called once per batch and per filter """
        pass

class EventFilterBase:
    def __init__(self, neededBranches=[]) -> None:
        self.neededBranches = neededBranches
        """ List of branches needed for the the filter (usually left empty) """

    
    def applyFilter(self, array:ak.Array, report:Report) -> ak.Array:
        return array

class NoFilter(EventFilterBase):
    """ Filter that does not filter anything """
    pass

class NumpyArrayFilter(EventFilterBase):
    def __init__(self, filterArray:np.ndarray[bool]) -> None:
        super().__init__()
        self.filterArray = filterArray

    def applyFilter(self, array: ak.Array, report:Report) -> ak.Array:
        filterArray_currentIteration = self.filterArray[report.global_entry_start:report.global_entry_stop]
        return array[filterArray_currentIteration]

class BaseComputation:
    eventFilter:EventFilterBase = NoFilter()
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
            computationTools = {compToolMaker : compToolMaker.instantiateComputationTool(array, report) for compToolMaker in computationToolMakers[eventFilter]}
            
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


