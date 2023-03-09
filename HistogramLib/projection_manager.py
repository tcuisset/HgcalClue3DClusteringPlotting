import hist
import dbm
from typing import List
import hist

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


class HistKindSelector:
    def registerCallback(self, callback):
        pass
    def getSelection(self) -> HistogramKind:
        return HistogramKind.COUNT

class HistogramProjectedView:
    store:HistogramStore
    myHist:MyHistogram
    projectedHist:hist.Hist
    histLambdaView = None # A lambda to apply to projectedHist that returns the relevant view for the current histogram kind
    histName:str
    shelfIdProviders:List[ShelfIdSelector]
    shelfId:ShelfId
    projectionProviders:dict

    plotAxises:List[str]|None = None
    histKindSelector:HistKindSelector

    def __init__(self, store:HistogramStore, shelfIdProviders:List[ShelfIdSelector], projectionProviders,
        histName:str, forcePlotAxis:List[str]|None=None, histKindSelector:HistKindSelector=HistKindSelector()) -> None:
        self.store = store
        self.myHist = None
        self.shelfIdProviders = shelfIdProviders
        self.projectionProviders = projectionProviders
        self.histName = histName
        self.plotAxises = forcePlotAxis
        self.histKindSelector = histKindSelector
        self.shelfId = ShelfId('default', 'data')
        self.callbacks = []
        
        for shelfIdProvider in self.shelfIdProviders:
            shelfIdProvider.registerCallback(self.updateShelf)
        for projectionProvider in self.projectionProviders.values():
            projectionProvider.registerCallback(self.updateProjection)
        self.histKindSelector.registerCallback(self.updateProjection)
        
        self.updateShelf(None, None, None)

    def registerUpdateCallback(self, callback):
        self.callbacks.append(callback)

    def updateShelf(self, attr, old, new):
        for shelfIdProvider in self.shelfIdProviders:
            shelfIdProvider.fillShelfId(self.shelfId)
        
        try:
            self.myHist = self.store.getShelf(self.shelfId)[self.histName]
        except dbm.error:
            if self.myHist is not None:
                self.myHist = self.myHist.getEmptyCopy() # make an empty copy so we keep the labels but remove the data
        self.updateProjection(None, None, None)

    def updateProjection(self, attr, old, new):
        slice_args_dict = {}
        unprojected_axes = []
        
        for axisName in self.myHist.axisNames():
            if axisName in self.projectionProviders:
                histogramSlice = self.projectionProviders[axisName].getSlice()
                if histogramSlice is not None:
                    slice_args_dict[axisName] = histogramSlice.getSliceObject()
            else:
                unprojected_axes.append(axisName)

        if self.plotAxises is None:
            # First call to updateProjection, figure out automatically the plotting axes and cache them
            self.plotAxises = unprojected_axes
        
        unprojected_hist, self.histLambdaView = self.myHist.getHistogramAndViewLambda(self.getHistKindSelection())
        # The project call might do nothing, but see ListHistogramSlice for why we should project all the time
        self.projectedHist = unprojected_hist[slice_args_dict].project(*self.plotAxises)
        for callback in self.callbacks:
            callback()
    
    def getHistKindSelection(self):
        """ Get the current histKind selection, potentially falling back on COUNT if requested is not available"""
        requestedKind = self.histKindSelector.getSelection()
        if self.myHist.hasHistogramType(requestedKind):
            return requestedKind
        else:
            return HistogramKind.COUNT

    def getHistogramBinCountLabel(self):
        return self.myHist.getHistogramBinCountLabel(self.getHistKindSelection())

    def getProjectedHistogramView(self):
        """ Get the view to plot, according to histogram kind selector"""
        return self.histLambdaView(self.projectedHist)

