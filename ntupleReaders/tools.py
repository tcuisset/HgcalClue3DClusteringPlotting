import awkward as ak

from ntupleReaders.computation import ComputationToolBase, ComputationToolMakerBase, Report
from hists.dataframe import DataframeComputations

class DataframeComputationsToolMaker(ComputationToolMakerBase):
    def __init__(self, neededBranches:list[str]=[], rechits_columns:list[str]=None) -> None:
        super().__init__(neededBranches=neededBranches)
        
        try:
            self.neededBranches.extend(rechits_columns)
        except TypeError:
            pass # in case rechits_columns is None

        self.rechits_columns = rechits_columns

    def instantiateComputationTool(self, array:ak.Array, report:Report) -> ComputationToolBase:
        """ Create a ComputationTool instance, called once per batch and per filter """
        if self.rechits_columns is None:
            return DataframeComputations(array)
        else:
            return DataframeComputations(array, rechits_columns=self.rechits_columns)

