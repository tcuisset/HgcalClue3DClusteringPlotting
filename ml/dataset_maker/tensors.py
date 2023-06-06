import typing
import os

import numpy as np
import awkward as ak
import uproot
import uproot.behaviors.TBranch
import torch

from ntupleReaders.computation import BaseComputation, Report, ComputationToolBase
from ntupleReaders.tools import DataframeComputationsToolMaker
from hists.dataframe import DataframeComputations

class TensorMakingComputation(BaseComputation):
    def __init__(self, *args, **kwargs) -> None:
        self.tensorFileName = kwargs.pop("tensorFileName")
        super().__init__(*args, **kwargs)
        self.tensorsList = []
        
        """ List of computed tensors to be merged """

    def getResult(self) -> torch.Tensor:
        """ Compute the full tensor"""
        try:
            return torch.cat(self.tensorsList)
        except TypeError:
            # in case we have tuples
            # first extract the list of tuples into tuple of lists
            tupleOfLists = zip(*self.tensorsList)
            # then concat each list individually, and tuple the tensors back
            return tuple(torch.cat(listOfTensors) for listOfTensors in tupleOfLists)
    
    def saveTensor(self, pathToFolder:str) -> None:
        """ Save the tensor to file """
        try:
            tensorName = self.tensorFileName
        except AttributeError:
            tensorName = self.__class__.__name__
        torch.save(self.getResult(), os.path.join(pathToFolder, tensorName + ".pt"))

class SimpleTensorMakingComputation(TensorMakingComputation):
    def __init__(self, **kwargs) -> None:
        super().__init__(neededComputationTools=[DataframeComputationsToolMaker(kwargs.pop("neededBranches"), rechits_columns=kwargs.pop("rechits_columns", None))], **kwargs)
    
    def computeTensor(self, comp:DataframeComputations) -> torch.Tensor:
        pass

    def processBatch(self, array:ak.Array, report:Report, computationTools:dict[typing.Type[ComputationToolBase], ComputationToolBase]) -> None:
        self.tensorsList.append(self.computeTensor(computationTools[DataframeComputations]))