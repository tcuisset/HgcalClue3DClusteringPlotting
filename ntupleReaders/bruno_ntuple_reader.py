import os
import sys
import functools
import pickle
import glob
from functools import cached_property

import uproot
import numpy as np

from HistogramLib.store import HistogramStore
from hists.store import HistogramId


class BrunoNtupleReader:
    def __init__(self, version, datatype) -> None:
        self.version = version
        self.datatype = datatype
        self.baseHistsFolder = "/grid_mnt/data_cms_upgrade/cuisset/testbeam18/ntuple-selection/"
        self.pathToFolder = os.path.join(self.baseHistsFolder, version)

    @cached_property
    def listOfNtuplesPaths(self) -> list[str]:
        return glob.glob(os.path.join(self.pathToFolder, f"ntuple_selection_{self.datatype}_em_*.root"))
    


