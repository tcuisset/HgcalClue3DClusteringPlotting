from dataclasses import dataclass
from collections import namedtuple
from typing import List
from enum import Enum, auto
import hist
import hist.axis
import hist.axestuple
import pandas as pd
import boost_histogram as bh

HistogramVariable = namedtuple('HistogramVariable', ['name', 'label'])



@dataclass
class HistogramSlice:
    axisName:str
    def getSliceObject(self):
        return None
    def shouldProjectLater(self):
        return False
    
    @property
    def legendLabel(self):
        return f"Projection on {self.axisName}"

@dataclass
class SingleValueHistogramSlice(HistogramSlice):
    value:str|int
    def getSliceObject(self):
        return hist.loc(self.value)
    @property
    def legendLabel(self):
        return str(self.value)

@dataclass
class RangeHistogramSlice(HistogramSlice):
    minRange:int # included
    maxRange:int # included 
    def getSliceObject(self):
        if self.maxRange == self.minRange:
            # Special case for single values (otherwise problems on first and last bins)
            return hist.loc(self.minRange)
        else:
            return slice(hist.loc(self.minRange), hist.loc(self.maxRange+1), sum) #slices are with end excluded so we add 1 to the max

@dataclass
class MinWithOverflowHistogramSlice(HistogramSlice):
    """ Slices the histogram from min (included) to the overflow bin (included)"""
    min:int
    def getSliceObject(self):
        return slice(hist.loc(self.min), None, sum)

@dataclass
class ListHistogramSlice(HistogramSlice):
    valueList:list
    def getSliceObject(self):
        # !!! does not project (needs to be projected)
        # for now boost-histogram using list slicing does not fill overflow bins
        # we need this behaviour for the project call in MultiBokehHistogram.getHistogramProjection to return the correct projection on selected beam energies
        # when boost-histogram is updated you should write probably [hist.loc(int(energy)) for energy in self.widget.value]::sum or something
        # see  https://github.com/scikit-hep/boost-histogram/issues/296
        return [hist.loc(value) for value in self.valueList]

    def shouldProjectLater(self):
        return True

@dataclass
class ProjectHistogramSlice(HistogramSlice):
    def shouldProjectLater(self):
        return True


class HistogramKind(Enum):
    COUNT = auto()
    WEIGHT = auto() # Histogram with Weight storage
    PROFILE = auto() # Histogram with Mean storage

@dataclass
class HistogramMetadata:
    title:str = ""
    binCountLabel:str = "Count" # Label that is plotted on y axis of 1D histogram (or colorbar of 2D) when profile is disabled. Usually "Event count".
    profileOn:HistogramVariable|None = None
    weightOn:HistogramVariable|None = None
    axes:hist.axestuple.NamedAxesTuple = hist.axestuple.NamedAxesTuple

    def getPlotLabel(self, kind:HistogramKind):
        """ Get the label of the bin contents of histogram, depending on profile (-> profile variable label) or count (-> stored in MyHistogram, usually "Event count")"""
        if kind is HistogramKind.COUNT or kind is None:
            return self.binCountLabel
        elif kind is HistogramKind.PROFILE:
            return self.profileOn.label
        elif kind is HistogramKind.WEIGHT:
            return self.weightOn.label
        else:
            raise ValueError("Wrong histogram kind", kind)


class MyHistogram():
    metadata:HistogramMetadata
    
    histDict:dict

    def __init__(self, *args, **kwargs) -> None:
        self.metadata = HistogramMetadata(
            title=kwargs.get('label', "Histogram"),
            binCountLabel=kwargs.pop('binCountLabel', "Count"),
            profileOn=kwargs.pop('profileOn', None),
            weightOn=kwargs.pop('weightOn', None)
        )

        self.histDict = {}
        
        #Note : Weight storage does not have a .count view parameter so we need to have a second histogram for counts
        # Mean storage however does have a .count view
        if self.metadata.profileOn is not None:
            # In case we want a profile histogram
            kwargs_profile = kwargs.copy()
            kwargs_profile["storage"] = hist.storage.Mean()
            self.histDict[HistogramKind.PROFILE] = hist.Hist(*args, **kwargs_profile)
        if self.metadata.weightOn is not None:
            # We want a weight histogram
            kwargs_weight = kwargs.copy()
            kwargs_weight["storage"] = hist.storage.Weight()
            self.histDict[HistogramKind.WEIGHT] = hist.Hist(*args, **kwargs_weight)
        if HistogramKind.COUNT not in self.histDict.keys():
            # We need a regular count histogram
            self.histDict[HistogramKind.COUNT] = hist.Hist(*args, **kwargs)
        
        self.metadata.axes = self.getHistogram(HistogramKind.COUNT).axes
 
    def getHistogram(self, kind:HistogramKind) -> hist.Hist:
        if kind in self.histDict:
            return self.histDict[kind]
        elif HistogramKind.PROFILE in self.histDict:
            return self.histDict[HistogramKind.PROFILE]
        else:
            raise ValueError("Histogram kind could not be found")
    
    def hasHistogramType(self, kind:HistogramKind) -> bool:
        return kind in self.histDict or (kind is HistogramKind.COUNT and HistogramKind.PROFILE in self.histDict)

    def getHistogramAndViewLambda(self, kind:HistogramKind):
        """ Get the histogram for the requested HistogramKind as well as a lambda to apply to the histogram to get the relevant view"""
        h:hist.Hist = self.getHistogram(kind)
        if kind is HistogramKind.PROFILE or kind is HistogramKind.WEIGHT:
            l = lambda h : h.view().value
        elif kind is HistogramKind.COUNT:
            if issubclass(h.storage_type, bh.storage.Mean):
                l = lambda h : h.view().count
            elif issubclass(h.storage_type, bh.storage.Weight) or issubclass(h.storage_type, bh.storage.WeightedMean):
                raise ValueError("Weight and WeightedMean storages do not have a count field")
            else:
                #Just assume the view is a "dumb" view with only counts
                l = lambda h : h.view()
        else:
            raise ValueError("Invalid HistogramKind") 
        return (h, l)

    @property
    def axes(self) -> hist.axis.NamedAxesTuple:
        return self.getHistogram(HistogramKind.COUNT).axes

    def fillFromDf(self, df:pd.DataFrame, mapping:dict={}, valuesNotInDf:dict={}):
        """
        mapping : dict hist_axis_name -> dataframe_axis_name
        valuesNotInDf : dict axis_name -> array of values or single value
            to be used if some columns are not in the dataframe, use the dict value then (directly passed to Hist.fill)
            (can also override the df column if it is the same name)
            these names are not mapped
        """
        dict_fill = {}
        
        def mapAxisName(axisName):
            return mapping[axisName] if axisName in mapping.keys() else axisName

        for ax in self.axes:
            if ax.name in valuesNotInDf:
                dict_fill[ax.name] = valuesNotInDf[ax.name]
            else:
                dict_fill[ax.name] = df[mapAxisName(ax.name)]

        for kind, h in self.histDict.items():
            if kind is HistogramKind.PROFILE:
                h.fill(**dict_fill, sample=df[mapAxisName(self.metadata.profileOn.name)])
            elif kind is HistogramKind.WEIGHT:
                h.fill(**dict_fill, weight=df[mapAxisName(self.metadata.weightOn.name)])
            elif kind is HistogramKind.COUNT:
                h.fill(**dict_fill)

    def getEmptyCopy(self):
        return MyHistogram(*self.axes)








