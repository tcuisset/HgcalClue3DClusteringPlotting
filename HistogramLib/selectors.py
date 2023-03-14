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

class Selection:
    label:str = "Selection (to be filled)"
    def slice(self) -> HistogramSlice|None:
        return None
    def histId(self) -> Tuple[str, str]|None:
        return None
    def histKind(self) -> HistogramKind|None:
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