import os
import sys
import functools
import pickle

import uproot
import numpy as np

from HistogramLib.store import HistogramStore
from hists.store import HistogramId


class ClueNtupleReader:
    def __init__(self, version, clueParams, datatype) -> None:
        self.version = version
        self.clueParams = clueParams
        self.datatype = datatype
        self.baseHistsFolder = "/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/"
        self.pathToFolder = os.path.join(self.baseHistsFolder, version, clueParams, datatype)
        """ Full path to folder holding CLUE_clusters.root """
        self.pathToFile = os.path.join(self.pathToFolder, "CLUE_clusters.root")
        self.pathToMLDatasetsFolder = os.path.join(self.pathToFolder, "ML_datasets")
        self.pathToSigmaOverEFolder = os.path.join(self.pathToFolder, "sigmaOverE")

        self.filterArrays:dict[str, np.ndarray[bool]] = dict()
    
    @functools.cached_property
    def tree(self) -> uproot.TTree:
        return uproot.open(self.pathToFile + ":clusters")
    
    def loadFilterArray(self, filterName="default_filter") -> np.ndarray[bool]:
        try:
            return self.filterArrays[filterName]
        except KeyError:
            self.filterArrays[filterName] = np.load(os.path.join(self.pathToFolder, "filters", filterName+".npy"))
            return self.filterArrays[filterName]

    @functools.cached_property
    def histStore(self) -> HistogramStore:
        return HistogramStore(os.path.join(self.baseHistsFolder, self.version), HistogramId)

    def loadSigmaOverEResults(self, resultName="rechits"):
        """ resultName can be : rechits, clue, clue3d """
        with open(os.path.join(self.pathToSigmaOverEFolder, resultName+".pickle"), 'rb') as f:
            try:
                return pickle.load(f)
            except (ModuleNotFoundError, AttributeError):
                from energy_resolution import sigma_over_e
                # hack to make notebook-saved pickle files work (from https://stackoverflow.com/a/2121918)
                sys.modules['sigma_over_e'] = sigma_over_e
                return pickle.load(f)
