from bokeh.layouts import row
import bokeh.models

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
        self.toggle = bokeh.models.Toggle(
            label="Explode",
        )
        self.model = row(baseSelector.model, self.toggle)
        self.callbacks = []
        self.toggle.on_change('active', self._toggleCallback)

    def selections(self) -> List[Selection]:
        if self.toggle.active:
            #Explode
            return self.baseSelector.allSelections
        else:
            return self.baseSelector.selections()

    def explodePlotType(self) -> ExplodePlotType:
        return self.baseSelector.explodePlotType()
    
    def registerCallback(self, callback):
        self.baseSelector.registerCallback(callback)
        self.callbacks.append(callback)

    def _toggleCallback(self, attr, old, new):
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
    def __init__(self, axisName:str, **kwargs) -> None:
        self.model = bokeh.models.RangeSlider(**kwargs)
        #Use value_throttled so that it updates only on mouse release to avoid recomputing all the time when dragging
        super().__init__(axisName, onChangeValue='value_throttled')
        self.allSelections = [SliceFixedSelection(SingleValueHistogramSlice(self.axisName, val)) for val in range(self.model.start, self.model.end+1, self.model.step)]

    def _updateSelection(self) -> None:
        (min, max) = self.model.value
        self.selection.sliceObject = RangeHistogramSlice(self.axisName, int(min), int(max))


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


