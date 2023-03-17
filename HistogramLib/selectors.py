from typing import List, Tuple
from enum import Enum, auto

from .histogram import HistogramSlice, HistogramKind

class ExplodePlotType(Enum):
    OVERLAY = auto() # Overlay on a single plot
    MULTIPLE_PLOT = auto() # Make separate plots

class SelectorType(Enum):
    HISTOGRAM_ID = auto() # Picks a histogram from file
    PROJECTION = auto()  # Chooses a projection
    HISTOGRAM_KIND = auto() # Count/Profile/Weight
    DENSITY_HISTOGRAM = auto() #Density/standard histogram

class Selection:
    label:str = "Selection (to be filled)"
    def slice(self) -> HistogramSlice|None:
        return None
    def histId(self) -> Tuple[str, str]|None:
        return None
    def histKind(self) -> HistogramKind|None:
        return None
    def densityHistogram(self) -> bool|None:
        """ Whether the histogram should be plotted as a density histogram or not """
        return None

class Selector:
    selectorType:SelectorType
    def selections(self) -> List[Selection]:
        pass
    def explodePlotType(self) -> ExplodePlotType:
        return ExplodePlotType.OVERLAY
    def registerCallback(self, callback):
        """  callback is a function : callback(triggerSelector:Selector, plotsHaveChanged=False)
        Pass plotsHaveChanged=False if len(selections()) is the same and Selection objects are the same (but possibly with different params)
        Pass plotsHaveChanged=True if len(selections()) has changed, or if the list contains altogether different Selection object """
        pass
    


class HistogramIdFixedSelection(Selection):
    def __init__(self, key, value=None) -> None:
        self.key = key
        self.value = value
        self.label = value
    def histId(self) -> Tuple[str, str] | None:
        return (self.key, self.value)

class HistogramIdNameSelector(Selector):
    def __init__(self, histName) -> None:
        self.selection = HistogramIdFixedSelection('histName', histName)
        self.selectorType = SelectorType.HISTOGRAM_ID
    def selections(self) -> List[Selection]:
        return [self.selection]

class HistogramIdNameMultiSelector(Selector):
    """ Select multiple (fixed) histogram names """
    def __init__(self, histNames:list[str]) -> None:
        self.selection_list = [HistogramIdFixedSelection('histName', histName) for histName in histNames]
        self.selectorType = SelectorType.HISTOGRAM_ID
    def selections(self) -> List[Selection]:
        return self.selection_list

class SliceFixedSelection(Selection):
    def __init__(self, slice:HistogramSlice=None) -> None:
        self.sliceObject = slice
    
    def slice(self) -> HistogramSlice | None:
        return self.sliceObject
    
    @property
    def label(self) -> str:
        return self.sliceObject.legendLabel if self.sliceObject is not None else "Projection on None axis"

class FixedSliceSelector(Selector):
    def __init__(self, slice) -> None:
        self.selection = SliceFixedSelection(slice)
        self.selectorType = SelectorType.PROJECTION
    def selections(self) -> List[Selection]:
        return [self.selections]


class HistKindFixedSelection(Selection):
    label = "Histogram kind selection"
    def __init__(self, kind:HistogramKind=HistogramKind.COUNT) -> None:
        self.kind = kind
    def histKind(self) -> HistogramKind | None:
        return self.kind

class FixedHistKindSelector(Selector):
    def __init__(self, kind:HistogramKind=HistogramKind.COUNT) -> None:
        self.selection = HistKindFixedSelection(kind)
        self.selectorType = SelectorType.HISTOGRAM_KIND
    def selections(self) -> List[Selection]:
        return [self.selection]

def findHistNameInSelections(selections:List[Selection]):
    for selection in selections:
        if selection.histId() is not None and selection.histId()[0] == "histName":
            return selection.histId()[1]
    raise RuntimeError("Histogram name was not found in Selection list")

class DensityHistogramFixedSelection(Selection):
    label = "Density selection"
    def __init__(self, density:bool) -> None:
        self.density = density
    def densityHistogram(self) -> bool | None:
        return self.density

class FixedDensityHistogramSelector(Selector):
    def __init__(self, density:bool) -> None:
        self.selection = DensityHistogramFixedSelection(density)
        self.selectorType = SelectorType.DENSITY_HISTOGRAM
    def selections(self) -> List[Selection]:
        return [self.selection]
