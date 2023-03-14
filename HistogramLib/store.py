from dataclasses import dataclass
import os
import glob
import pickle
import copy

from .histogram import HistogramMetadata

@dataclass(unsafe_hash=True)
class ShelfId:
    clue_param_name:str
    datatype:str
    shelfFileName:str = 'hists.shelf'

    @property
    def path(self):
        return os.path.join(self.clue_param_name, self.datatype, self.shelfFileName)

class AbstractHistogramId:
    histName:str = ""
    @property
    def path(self) -> str:
        """ get relative path from version folder (excluded) to pickle file name (included)"""
        return ""


def discoverShelves(hist_folder):
    shelfIdList = []
    #                           clueParams/datatype/hists.shelf
    paths = glob.glob(os.path.join('*',      '*',  'hists.shelf.*'), root_dir=hist_folder)
    for path in paths:
        folderPath, _ = os.path.split(path) # clueParams/datatype
        clueParam, datatype = os.path.split(folderPath)
        shelfIdList.append(ShelfId(clue_param_name=clueParam, datatype=datatype))
    return shelfIdList


class HistogramStore:
    def __init__(self, hist_folder, histIdClass) -> None:
        """
        hist_folder is the path to the tag folder (included), ie it has to be a folder where there is clue_param_name/datatype/... subdirectory and files
        histIdClass is the *class* (not object) to use for indexing histograms, should implement the interface AbstractHistogramId
        """
        self.loadedHists = {}
        self.metadataDict = {}
        self.hist_folder = hist_folder
        self.histIdClass = histIdClass

    def get(self, id:AbstractHistogramId):
        if id not in self.loadedHists:
            self._loadHist(id)
        return self.loadedHists[id] 

    def getMetadata(self, histName:str) -> HistogramMetadata:
        if not self.metadataDict: # Empty dict
            with open(os.path.join(self.hist_folder, "metadata.pickle"), "rb") as f:
                self.metadataDict = pickle.load(f)
        return self.metadataDict[histName]

    def save(self, id:AbstractHistogramId, hist, makedirs=True, saveMetadata=False):
        pathToFile = os.path.join(self.hist_folder, id.path)
        if makedirs:
            os.makedirs(os.path.dirname(pathToFile), exist_ok=True)
        with open(pathToFile, mode='wb') as f:
            pickle.dump(hist, f)
        if saveMetadata:
            self.metadataDict[id.histName] = hist.metadata
    
    def saveMetadataDict(self):
        with open(os.path.join(self.hist_folder, "metadata.pickle"),  mode='wb') as f:
            pickle.dump(self.metadataDict, f)

    def _loadHist(self, id:AbstractHistogramId):
        # Copy the id otherwise it could be changed by the caller and break the dictionnary
        with open(os.path.join(self.hist_folder, id.path), "rb") as f:
            self.loadedHists[copy.copy(id)] = pickle.load(f)
    
    def getPossibleClueParameters(self):
        #                           clueParams/datatype/histogram_name.pickle
        paths = glob.glob(os.path.join('*',      '*',  '*.pickle'), root_dir=self.hist_folder)
        clueParams = set() # Make a set to remove duplicates
        for path in paths:
            folderPath, _ = os.path.split(path) # clueParams/datatype
            clueParam, datatype = os.path.split(folderPath)
            clueParams.add(clueParam)
        return list(clueParams)
