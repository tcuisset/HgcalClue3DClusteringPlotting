import hist
import dbm
from typing import List
import hist

from .store import *
from .histogram import *
from .selectors import *


class HistogramLoadError(Exception):
    pass

class HistogramView:
    def __init__(self, store:HistogramStore, selections:List[Selection], forcePlotAxises:List[str]|None=None) -> None:
        self.store:HistogramStore = store
        self.selections = selections
        self.forcePlotAxises = forcePlotAxises
        self._reset()

        self.update()

    def _reset(self):
        self.myHist:MyHistogram = None
        self.projectedHist:hist.Hist = None
        self.histId:AbstractHistogramId = None
        self.histKind:HistogramKind = None
        self.histLambdaView = None # A lambda to apply to projectedHist that returns the relevant view for the current histogram kind
        self.slice_args_dict = None

    def _resetProjection(self):
        self.myHist = self.myHist.getEmptyCopy() # make an empty copy so we keep the labels but remove the data

    def _buildParams(self):
        histId = self.store.histIdClass()
        slice_args_dict = {}
        histogramKind = None
        for selection in self.selections:
            histIdTuple = selection.histId()
            slice = selection.slice()
            
            if histIdTuple is not None:
                setattr(histId, histIdTuple[0], histIdTuple[1])
            elif slice is not None:
                if slice.axisName in slice_args_dict:
                    raise ValueError("Cannot specify multiple times the same axisName to project on")
                slice_args_dict[slice.axisName] = slice.getSliceObject()
            elif selection.histKind() is not None:
                if histogramKind is not None:
                    raise ValueError("Cannot specify multiple times HistogramKind")
                histogramKind = selection.histKind()
        return histId, slice_args_dict, histogramKind
    
    def _updateHist(self, newHistId):
        self.myHist = self.store.get(newHistId)
        self.histId = newHistId

    def _updateProjection(self):
        if self.forcePlotAxises is not None:
            plotAxises = self.forcePlotAxises
        else:
            plotAxises = []
        
        new_slice_args_dict = {} #Subset with only the axises that are in our histogram
        for axisName in self.myHist.axes.name:
            if self.forcePlotAxises is None and axisName not in self.slice_args_dict:
                # Figure out automatically the plotting axes 
                plotAxises.append(axisName)
            if axisName in self.slice_args_dict:
                new_slice_args_dict[axisName] = self.slice_args_dict[axisName]
        
        unprojected_hist, self.histLambdaView = self.myHist.getHistogramAndViewLambda(self.histKind)
        # The project call might do nothing, but see ListHistogramSlice for why we should project all the time
        self.projectedHist = unprojected_hist[new_slice_args_dict].project(*plotAxises)

    def update(self):
        histId, slice_args_dict, requestedKind = self._buildParams()
        forceNext = False
        try:
            self.exception = None
            if histId != self.histId:
                self._updateHist(histId)
                forceNext = True
        except Exception as e:
            self.exception = e
            self._reset()
        else:
            if self.myHist.hasHistogramType(requestedKind):
                newHistKind = requestedKind
            else:
                newHistKind = HistogramKind.COUNT # Fallback on COUNT if we selected Weight or Profile and it's not available

            if slice_args_dict != self.slice_args_dict or newHistKind != self.histKind or forceNext:
                self.histKind = newHistKind
                self.slice_args_dict = slice_args_dict
                self._updateProjection()


    def getProjectedHistogramView(self):
        """ Get the view to plot, according to histogram kind selector"""
        if self.projectedHist is not None:
            return self.histLambdaView(self.projectedHist)
        else:
            raise HistogramLoadError(self.exception)

