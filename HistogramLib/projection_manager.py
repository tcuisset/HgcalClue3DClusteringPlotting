import hist
import dbm
from typing import List
import hist

from .store import *
from .histogram import *

class HistogramIdSelector:
    def fillHistId(self, histId:AbstractHistogramId):
        pass
    def registerCallback(self, callback):
        pass
class HistogramIdNameSelector(HistogramIdSelector):
    def __init__(self, histName) -> None:
        self.histName = histName
    def fillHistId(self, histId):
        histId.histName = self.histName

class ProjectionAxisSelector:
    axisName:str
    def getSlice(self) -> HistogramSlice:
        return None
    def registerCallback(self, callback):
        pass

class PlaceholderAxisSelector(ProjectionAxisSelector):
    def __init__(self, slice) -> None:
        self.slice = slice
    def getSlice(self):
        return self.slice

class HistKindSelector:
    def registerCallback(self, callback):
        pass
    def getSelection(self) -> HistogramKind:
        return HistogramKind.COUNT

class FixedHistKindSelector(HistKindSelector):
    def __init__(self, kind:HistogramKind) -> None:
        self.kind = kind
    def getSelection(self) -> HistogramKind:
        return self.kind

class HistogramProjectedView:
    store:HistogramStore
    myHist:MyHistogram
    projectedHist:hist.Hist
    histLambdaView = None # A lambda to apply to projectedHist that returns the relevant view for the current histogram kind
    histIdProviders:List[HistogramIdSelector]
    projectionProviders:dict

    plotAxises:List[str]|None = None
    histKindSelector:HistKindSelector

    def __init__(self, store:HistogramStore, histIdProviders:List[HistogramIdSelector], projectionProviders,
        forcePlotAxis:List[str]|None=None, histKindSelector:HistKindSelector=HistKindSelector()) -> None:
        self.store = store
        self.myHist = None
        self.histIdProviders = histIdProviders
        self.projectionProviders = projectionProviders
        self.plotAxises = forcePlotAxis
        self.histKindSelector = histKindSelector
        self.histId = ShelfId('default', 'data')
        self.callbacks = []
        self
        
        for histIdProvider in self.histIdProviders:
            histIdProvider.registerCallback(self.updateHistogram)
        for projectionProvider in self.projectionProviders.values():
            projectionProvider.registerCallback(self.updateProjection)
        self.histKindSelector.registerCallback(self.updateProjection)
        
        self.updateHistogram(None, None, None)

    def registerUpdateCallback(self, callback):
        self.callbacks.append(callback)

    def updateHistogram(self, attr, old, new):
        histId = self.store.histIdClass()
        for histIdProvider in self.histIdProviders:
            histIdProvider.fillHistId(histId)
        
        newHist = None
        try:
            newHist = self.store.get(histId)
        except Exception as e:
            print(e)
        
        if newHist is None and self.myHist is not None:
            self.myHist = self.myHist.getEmptyCopy() # make an empty copy so we keep the labels but remove the data
        else:
            self.myHist = newHist
        
        self.updateProjection(None, None, None)

    def updateProjection(self, attr, old, new):
        if self.myHist is not None:
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
        else:
            self.projectedHist = None
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

