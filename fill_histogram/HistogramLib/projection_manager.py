import hist
from typing import List

from .store import *
from .histogram import *

class ShelfIdSelector:
    def fillShelfId(self, shelfId:ShelfId):
        pass
    def registerCallback(self, callback):
        pass
class ProjectionAxisSelector:
    axisName:str
    def getSlice(self) -> HistogramSlice:
        return None
    def registerCallback(self, callback):
        pass

class ToggleProfile:
    def registerCallback(self, callback):
        pass
    def shouldProfile(self) -> bool:
        return False

class HistogramProjectedView:
    store:HistogramStore
    hist:MyHistogram
    projectedHist:MyHistogram
    histName:str
    shelfIdProviders:List[ShelfIdSelector]
    shelfId = ShelfId('default', 'data')
    projectionProviders:dict

    plotAxises:List[str]|None = None
    toggleProfile:ToggleProfile

    def __init__(self, store:HistogramStore, shelfIdProviders:List[ShelfIdSelector], projectionProviders,
        histName:str, forcePlotAxis:List[str]|None=None, toggleProfileButton:ToggleProfile=None) -> None:
        self.store = store
        self.shelfIdProviders = shelfIdProviders
        self.projectionProviders = projectionProviders
        self.histName = histName
        self.plotAxises = forcePlotAxis
        self.toggleProfile = toggleProfileButton
        
        for shelfIdProvider in self.shelfIdProviders:
            shelfIdProvider.registerCallback(self.updateShelf)
        for projectionProvider in self.projectionProviders.values():
            projectionProvider.registerCallback(self.updateProjection)
        
        self.updateShelf(None, None, None)

    def registerUpdateCallback(self, callback):
        for shelfIdProvider in self.shelfIdProviders:
            shelfIdProvider.registerCallback(callback)
        for projectionProvider in self.projectionProviders.values():
            projectionProvider.registerCallback(callback)
        self.toggleProfile.registerCallback(callback)

    def updateShelf(self, attr, old, new):
        for shelfIdProvider in self.shelfIdProviders:
            shelfIdProvider.fillShelfId(self.shelfId)
        self.hist = self.store.getShelf(self.shelfId)[self.histName]
        self.updateProjection(None, None, None)

    def updateProjection(self, attr, old, new):
        slice_args_dict = {}
        unprojected_axes = []
        
        for axisName in self.hist.axisNames():
            if axisName in self.projectionProviders:
                histogramSlice = self.projectionProviders[axisName].getSlice()
                if histogramSlice is not None:
                    slice_args_dict[axisName] = histogramSlice.getSliceObject()
            else:
                unprojected_axes.append(axisName)

        if self.plotAxises is None:
            # First call to updateProjection, figure out automatically the plotting axes and cache them
            self.plotAxises = unprojected_axes
        
        # This might do nothing, but see ListHistogramSlice for why we should project all the time
        self.projectedHist = self.hist[slice_args_dict].project(*self.plotAxises)
    
    def getProjectedHistogramView(self):
        """ Get the view to plot, according to toggleProfile"""
        if self.toggleProfile.shouldProfile() and self.hist.isProfile():
            return self.projectedHist.view().value
        elif self.hist.isProfile():
            return self.projectedHist.view().count
        else:
            return self.projectedHist.view()
