from dataclasses import dataclass
from collections import namedtuple
from typing import List
import hist
import hist.axis
import pandas as pd
import boost_histogram as bh

ProfileVariable = namedtuple('ProfileVariable', ['name', 'label'])



@dataclass
class HistogramSlice:
    axisName:str
    def getSliceObject(self):
        return None
    def shouldProjectLater(self):
        return False

@dataclass
class SingleValueHistogramSlice(HistogramSlice):
    value:str|int
    def getSliceObject(self):
        return hist.loc(self.value)

@dataclass
class RangeHistogramSlice(HistogramSlice):
    minRange:int
    maxRange:int
    def getSliceObject(self):
        return slice(hist.loc(self.minRange), hist.loc(self.maxRange+1), sum)

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




class MyHistogram(hist.Hist, family=None): # see https://hist.readthedocs.io/en/latest/user-guide/subclassing.html
    binCountLabel:str = "Count" # Label that is plotted on y axis of 1D histogram (or colorbar of 2D) when profile is disabled. Usually "Event count".
    profileOn:ProfileVariable|None = None

    def __init__(self, *args, **kwargs) -> None:
        self.profileOn = kwargs.pop('profileOn', None)
        if self.profileOn is not None and "storage" not in kwargs:
            # In case we want a profile histogram
            kwargs["storage"] = hist.storage.Mean()
        self.binCountLabel = kwargs.pop('binCountLabel', self.binCountLabel)
        super().__init__(*args, **kwargs)
    
    def isProfile(self):
        #return self.profileOn is not None
        return self.kind is not bh.Kind.COUNT

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
        
        if self.profileOn is not None:
            sample = df[mapAxisName(self.profileOn.name)]
        else:
            sample = None

        self.fill(**dict_fill, sample=sample)

    def axisNames(self):
        return [axis.name for axis in self.axes]

    def getEmptyCopy(self):
        return MyHistogram(*self.axes)







