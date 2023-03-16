from typing import List, Tuple, Type, Callable
from collections.abc import Iterable
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
            singlePlotClass:Type[AbstractPlotClass], multiPlotClass:Type[AbstractPlotClass]|None=None,
            lambdaOnPlotCreation:Callable[[AbstractPlotClass], None]=None) -> None:
        """ lambdaOnPlotCreation is a facultative lambda/function that is called each time a PlotClass is instantiated, with the PlotClass in parameter"""
        self.store = store
        self.singlePlotClass = singlePlotClass
        self.multiPlotClass = multiPlotClass
        self.selectors = selectors
        self.lambdaOnPlotCreation = lambdaOnPlotCreation
        self.model = Row()
        self.plots:List[AbstractPlotClass] = []

        for selector in selectors:
            selector.registerCallback(self.onSelectorCallback)
        self.makeEverything()

    def _explode(self) -> Tuple[Selector, Iterable[tuple[tuple[Selection, int], ...]]]:
        """ Return a all possible selections to plot according to the explode button
        
        Returns: tuple (overlaySelector, allSelections) where
         - overlaySelector is a Selector, whose selections are to be plotted as an overlay plot (eventually None if no overlay is to be used)
         - allSelections is a generator of tuples of tuples, each sub-tuple being of the form (Selection, lenSelections)
             where lenSelections is the number of different Selection objects returned by the corresponding Selector
             (1 -> no exploding of plots, 3 -> for example, result of data, sim_proton, sim_noproton being exploded)
         """
        selectionsToExplode = []
        overlaySelector = None
        for selector in self.selectors:
            if self.multiPlotClass is not None and overlaySelector is None and len(selector.selections()) > 1 and selector.explodePlotType() is ExplodePlotType.OVERLAY:
                overlaySelector = selector
            else:
                selectionsToExplode.append([(selection, len(selector.selections())) for selection in selector.selections()])
        allSelections = product(*selectionsToExplode) # cartesian product
        return (overlaySelector, allSelections)
    
    def _updateModel(self):
        self.model.children = [plot.figure for plot in self.plots]

    def makeMetadata(self, selectionLengthTuple:Iterable[Tuple[Selection, int]]):
        metadata = copy.copy(self.store.getMetadata(findHistNameInSelections([selection for selection, lenSelections in selectionLengthTuple])))
        slicedAxes = set()
        for selection, lenSelections in selectionLengthTuple:
            if selection.slice() is not None and selection.slice().axisName in metadata.axes.name:
                slicedAxes.add(selection.slice().axisName)
        metadata.axes = NamedAxesTuple(metadata.axes[axisName] for axisName in set(metadata.axes.name).difference(slicedAxes))
        metadata.title += " "+" ".join(selection.label for selection, lenSelections in selectionLengthTuple if lenSelections > 1)
        return metadata

    def makeEverything(self):
        overlaySelector, allSelections = self._explode()
        self.plots = []
        firstFigure:figure = None
        for selectionAndLengthTuple in allSelections: #selectionAndLengthTuple is tuple[tuple[Selection, int], ...]]
            selectionList = [selection for selection, lenSelections in selectionAndLengthTuple]
            if overlaySelector is not None:
                #Use for metadata the first overlay (otherwise the overlay axis is kept in metadata which is not wanted)
                # TODO : have overlay axis in metadata

                newPlot = self.multiPlotClass(
                    # Append to selectionAndLengthTuple an element (with lenSelections=1)
                    metadata=self.makeMetadata(selectionAndLengthTuple+((overlaySelector.selections()[0], 1),)),
                    projectedViews={
                        overlaySelection.label : HistogramView(
                            self.store,
                            list(selectionList+[overlaySelection])) # add overlay to tuple 
                        for overlaySelection in overlaySelector.selections()
                    }
                )
            else:
                figure_kwargs = {}
                if firstFigure is not None:
                    # Enable linked panning and zooming : https://docs.bokeh.org/en/latest/docs/user_guide/interaction/linking.html#linked-panning
                    figure_kwargs["x_range"] = firstFigure.x_range
                    figure_kwargs["y_range"] = firstFigure.y_range
                
                newPlot = self.singlePlotClass(metadata=self.makeMetadata(selectionAndLengthTuple),
                    projectedView=HistogramView(
                        self.store,
                        selectionList
                    ),
                    **figure_kwargs
                )
                if firstFigure is None:
                    firstFigure = newPlot.figure

            self.plots.append(newPlot)
            if self.lambdaOnPlotCreation is not None:
                self.lambdaOnPlotCreation(newPlot)
        self._updateModel()

    def onSelectorCallback(self, triggerSelector:Selector, plotsHaveChanged=False):
        """ plotsHaveChanged : set to True if the number of plots has changed (ie len(Selector.selections()) has changed)"""
        triggerSelectorType = triggerSelector.selectorType
        if triggerSelectorType is SelectorType.HISTOGRAM_ID or plotsHaveChanged:
            # If hitogramId have changed we need to load new histograms from files
            # If plotsHaveChanhed=True we have new plots so in this case just update everything
            self.makeEverything()
        elif triggerSelectorType in [SelectorType.PROJECTION, SelectorType.HISTOGRAM_KIND, SelectorType.DENSITY_HISTOGRAM]:
            # Only projection values have changed
            for plot in self.plots:
                plot.update()
        else:
            raise ValueError("SelctorType not supported")