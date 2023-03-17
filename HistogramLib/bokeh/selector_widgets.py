import math
from bokeh.layouts import row
import bokeh.models
from bokeh.models.formatters import CustomJSTickFormatter

from ..histogram import *
from ..projection_manager import *
from ..store import *
from ..selectors import *

class ExplodableSelector:
    #model:bokeh.models.Model
    allSelections:List[Selection]

class ExplodeSelector(Selector):
    def __init__(self, baseSelector:ExplodableSelector) -> None: #baseSelector must be Selector and have a .model and .allSelections
        self.baseSelector = baseSelector
        self.selectorType = baseSelector.selectorType
        self.button = bokeh.models.RadioButtonGroup(
            labels=["Single", "Duplicate", "Overlay"],
            active=0
        )
        self.model = row(baseSelector.model, self.button)
        self.callbacks = []
        self.button.on_event('button_click', self._buttonCallback)

    def selections(self) -> List[Selection]:
        if self.button.active == 1 or self.button.active == 2:
            #Explode
            return self.baseSelector.allSelections
        else:
            return self.baseSelector.selections()

    def explodePlotType(self) -> ExplodePlotType:
        if self.button.active == 1:
            return ExplodePlotType.MULTIPLE_PLOT
        else:
            return ExplodePlotType.OVERLAY
    
    def registerCallback(self, callback):
        self.baseSelector.registerCallback(callback)
        self.callbacks.append(callback)

    def _buttonCallback(self):
        for callback in self.callbacks:
            callback(self, plotsHaveChanged=True)

class ProjectionSelectorImpl(Selector):
    selectorType = SelectorType.PROJECTION
    def __init__(self, axisName:str, onChangeValue='value') -> None:
        self.axisName = axisName
        self.selection = SliceFixedSelection()
        self._updateSelection()
        self.model.on_change(onChangeValue, self._modelCallback)
        self.callbacks = []

    def _modelCallback(self, attr, old, new):
        self._updateSelection()
        for callback in self.callbacks:
            callback(self)
    
    def selections(self) -> List[Selection]:
        return [self.selection]

    def registerCallback(self, callback):
        self.callbacks.append(callback)

class RangeAxisSelector(ProjectionSelectorImpl, ExplodableSelector):
    """ 
    logScale : enable logScale of slider. It works by changing slider values to the log of the value, 
    and then doing exp() when showing the value
    """
    def __init__(self, axisName:str, logScale=False, **kwargs) -> None:
        self.logScale = logScale
        if logScale:
            kwargs["start"] = math.log(kwargs["start"])
            kwargs["end"] = math.log(kwargs["end"])
            kwargs["value"] = (math.log(kwargs["value"][0]), math.log(kwargs["value"][1]))
            kwargs["format"] = CustomJSTickFormatter(code="return Math.exp(tick).toFixed(2)") # Show the regular (non-log) value on the slider
        self.model = bokeh.models.RangeSlider(**kwargs)
        #Use value_throttled so that it updates only on mouse release to avoid recomputing all the time when dragging
        super().__init__(axisName, onChangeValue='value_throttled')
    
    @property
    def allSelections(self):
        #Note this does not work with float values for start, stop, step, in this case this should be changed to np.arange
        return [SliceFixedSelection(SingleValueHistogramSlice(self.axisName, val)) for val in range(self.model.start, self.model.end+1, self.model.step)]

    def _updateSelection(self) -> None:
        (min, max) = self.model.value
        if self.logScale:
            min = math.exp(min)
            max = math.exp(max)
        self.selection.sliceObject = RangeHistogramSlice(self.axisName, min, max)


class MultiSelectAxisSelector(ProjectionSelectorImpl, ExplodableSelector):
    def __init__(self, axisName:str, **kwargs) -> None:
        self.model = bokeh.models.MultiSelect(**kwargs)
        super().__init__(axisName)
        self.allSelections = [SliceFixedSelection(SingleValueHistogramSlice(self.axisName, int(val[0]))) for val in self.model.options]

    def _updateSelection(self) -> None:
        self.selection.sliceObject = ListHistogramSlice(self.axisName, [int(val) for val in self.model.value])

class RadioButtonGroupAxisSelector(ProjectionSelectorImpl, ExplodableSelector):
    """ You need to pass labels as kwargs with the same labels as category axis values """
    def __init__(self, axisName:str, **kwargs) -> None:
        self.model = bokeh.models.RadioButtonGroup(**kwargs)
        super().__init__(axisName, onChangeValue='active')
        self.allSelections = [SliceFixedSelection(SingleValueHistogramSlice(self.axisName, label)) for label in self.model.labels]
    
    def _updateSelection(self) -> None:
        self.selection.sliceObject = SingleValueHistogramSlice(self.axisName, self.model.labels[self.model.active])

class SliderMinWithOverflowAxisSelector(ProjectionSelectorImpl):
    """ Projects on [value:] with upper overflow bin included (undeflow excluded)"""
    def __init__(self, axisName:str, **kwargs) -> None:
        self.model = bokeh.models.Slider(**kwargs)
        super().__init__(axisName, onChangeValue='value_throttled')

    def _updateSelection(self) -> None:
        self.selection.sliceObject = MinWithOverflowHistogramSlice(self.axisName, int(self.model.value))


