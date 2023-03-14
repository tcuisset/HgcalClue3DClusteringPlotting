from typing import List, Tuple, Type
from itertools import product
import copy

from hist.axestuple import NamedAxesTuple
from bokeh.models import Row
from bokeh.plotting import figure

from HistogramLib.selectors import *
from HistogramLib.projection_manager import *


class AbstractPlotClass:
    #figure:Figure
    def update(self) -> None:
        pass

class PlotManager:
    def __init__(self, store:HistogramStore,selectors:List[Selector],
            singlePlotClass:Type[AbstractPlotClass], multiPlotClass:Type[AbstractPlotClass]|None=None) -> None:
        self.store = store
        self.singlePlotClass = singlePlotClass
        self.multiPlotClass = multiPlotClass
        self.selectors = selectors
        self.model = Row()
        self.plots:List[AbstractPlotClass] = []

        for selector in selectors:
            selector.registerCallback(self.onSelectorCallback)
        self.makeEverything()

    def _explode(self):
        selectionsToExplode = []
        overlaySelector = None
        for selector in self.selectors:
            if self.multiPlotClass is not None and overlaySelector is None and len(selector.selections()) > 1 and selector.explodePlotType() is ExplodePlotType.OVERLAY:
                overlaySelector = selector
            else:
                selectionsToExplode.append(selector.selections())
        allSelections = product(*selectionsToExplode) # cartesian product
        return (overlaySelector, allSelections)
    
    def _updateModel(self):
        self.model.children = [plot.figure for plot in self.plots]

    def makeMetadata(self, selections):
        metadata = copy.copy(self.store.getMetadata(findHistNameInSelections(selections)))
        slicedAxes = set()
        for selection in selections:
            if selection.slice() is not None and selection.slice().axisName in metadata.axes.name:
                slicedAxes.add(selection.slice().axisName)
        metadata.axes = NamedAxesTuple(metadata.axes[axisName] for axisName in set(metadata.axes.name).difference(slicedAxes))
        return metadata

    def makeEverything(self):
        overlaySelector, allSelections = self._explode()
        self.plots = []
        firstFigure:figure = None
        for selectionTuple in allSelections:
            if overlaySelector is not None:
                #Use for metadata the first overlay (otherwise the overlay axis is kept in metadata which is not wanted)
                # TODO : have overlay axis in metadata
                self.plots.append(self.multiPlotClass(metadata=self.makeMetadata(selectionTuple+(overlaySelector.selections()[0],)),
                    projectedViews={
                        overlaySelection.label : HistogramView(
                            self.store,
                            list(selectionTuple+(overlaySelection,))) # add overlay to tuple 
                        for overlaySelection in overlaySelector.selections()
                    }
                ))
            else:
                figure_kwargs = {}
                if firstFigure is not None:
                    # Enable linked panning and zooming : https://docs.bokeh.org/en/latest/docs/user_guide/interaction/linking.html#linked-panning
                    figure_kwargs["x_range"] = firstFigure.x_range
                    figure_kwargs["y_range"] = firstFigure.y_range
                
                self.plots.append(self.singlePlotClass(metadata=self.makeMetadata(selectionTuple),
                    projectedView=HistogramView(
                        self.store,
                        list(selectionTuple)
                    ),
                    **figure_kwargs
                ))
                if firstFigure is None:
                    firstFigure = self.plots[0].figure
        
        self._updateModel()

    def onSelectorCallback(self, triggerSelector:Selector, plotsHaveChanged=False):
        """ plotsHaveChanged : set to True if the number of plots has changed (ie len(Selector.selections()) has changed)"""
        triggerSelectorType = triggerSelector.selectorType
        if triggerSelectorType is SelectorType.HISTOGRAM_ID or plotsHaveChanged:
            # If hitogramId have changed we need to load new histograms from files
            # If plotsHaveChanhed=True we have new plots so in this case just update everything
            self.makeEverything()
        elif triggerSelectorType is SelectorType.PROJECTION or  triggerSelectorType is SelectorType.HISTOGRAM_KIND:
            # Only projection values have changed
            for plot in self.plots:
                plot.update()
        else:
            raise ValueError("SelctorType not supported")