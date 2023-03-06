from dataclasses import dataclass
import os
import glob
import shelve
import copy

@dataclass(unsafe_hash=True)
class ShelfId:
    clue_param_name:str
    datatype:str
    shelfFileName:str = 'hists.shelf'

    @property
    def path(self):
        return os.path.join(self.clue_param_name, self.datatype, self.shelfFileName)

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
    """
    clue_param_name / datatype / hists.shelf ->  hist_name : MyHistogram
    """
    openShelves = {}
    dbmFlag:str
    makeDirs:bool

    def __init__(self, hist_folder, dbmFlag='r', makedirs=False) -> None:
        """
        hist_folder is the path to the tag folder, ie it has to be a folder where there is clue_param_name/datatype/hists.shelf.* subsirectory and files
        dbmFlag meaning  :
        'r' Open existing database for reading only (default)
        'w' Open existing database for reading and writing
        'c' Open database for reading and writing, creating it if it doesn’t exist
        'n' Always create a new, empty database, open for reading and writing
        """
        self.hist_folder = hist_folder
        self.dbmFlag = dbmFlag
        self.makeDirs = makedirs
        pass

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self._closeAllShelves()
        pass

    def _openShelf(self, shelfId:ShelfId):
        """
        Precondition: shelf not already open
        raises dbm.error
        """
        # if shelfId in self.openShelves:
        #     raise ValueError("Shelf already open " + repr(shelfId))
        pathToFile = os.path.join(self.hist_folder, shelfId.path)
        if self.makeDirs:
            os.makedirs(os.path.dirname(pathToFile), exist_ok=True)
        # We need to copy shelfId
        self.openShelves[copy.copy(shelfId)] = shelve.open(pathToFile, flag=self.dbmFlag)

    def _closeAllShelves(self):
        for shelf in self.openShelves.values():
            shelf.close()

    def getShelf(self, shelfId:ShelfId):
        if shelfId not in self.openShelves:
            self._openShelf(shelfId)
        return self.openShelves[shelfId]
    
    def getPossibleClueParameters(self):
        # Make a set to remove duplicates
        return list({shelfId.clue_param_name for shelfId in discoverShelves(self.hist_folder)})

