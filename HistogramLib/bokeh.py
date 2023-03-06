import bokeh.models
import bokeh.plotting
import boost_histogram as bh
import numpy as np
from bokeh.models import ColumnDataSource

from .histogram import *
from .projection_manager import *
from .store import *


class RangeAxisSelector(ProjectionAxisSelector):
    def __init__(self, axisName:str, **kwargs) -> None:
        self.axisName = axisName
        self.widget = bokeh.models.RangeSlider(**kwargs)

    def getSlice(self):
        (min, max) = self.widget.value
        return RangeHistogramSlice(self.axisName, int(min), int(max)+1)
    
    def registerCallback(self, callback):
        #Use value_throttled so that it updates only on mouse release to avoid recomputing all the time when dragging
        self.widget.on_change('value_throttled', callback)

class MultiSelectAxisSelector(ProjectionAxisSelector):
    def __init__(self, axisName:str, **kwargs) -> None:
        self.axisName = axisName
        self.widget = bokeh.models.MultiSelect(**kwargs)
    
    def getSlice(self):
        return ListHistogramSlice(self.axisName, [int(val) for val in self.widget.value])
    def registerCallback(self, callback):
        self.widget.on_change('value', callback)

class PlaceholderAxisSelector(ProjectionAxisSelector):
    def __init__(self, slice) -> None:
        self.slice = slice
    def getSlice(self):
        return self.slice



class BokehHistogram:
    def __init__(self, histProvider:HistogramProjectedView, **kwargs_figure) -> None:
        if len(histProvider.projectedHist.axes) != 1:
            raise ValueError("You are trying to plot a 1D histogram using multidimensional data, you are missing a projection. Histogram axes : " 
                + ", ".join(histProvider.projectedHist.axisNames))

        self.figure = bokeh.plotting.figure(**kwargs_figure)
        self.histProvider = histProvider
        
        self.figure.title = histProvider.hist.label
        self.figure.xaxis.axis_label = self.histProvider.projectedHist.axes[0].label
        self.figure.yaxis.axis_label = self.histProvider.hist.profileOn.label if self.histProvider.hist.isProfile() else "Count"

        self.source = ColumnDataSource()
        self.update(None, None, None)
        self.figure.quad(bottom=0, source=self.source)

        self.histProvider.registerUpdateCallback(self.update)
    
    def update(self, attr, old, new):
        h_proj = self.histProvider.projectedHist
        self.source.data = {"top":self.histProvider.getProjectedHistogramView(), "left":h_proj.axes[0].edges[:-1], "right":h_proj.axes[0].edges[1:]}



class MultiBokehHistogram2D:
    def __init__(self, histProvider:HistogramProjectedView, **kwargs) -> None:
        if len(histProvider.projectedHist.axes) != 2:
            raise ValueError("You are trying to plot a 2D histogram with the wrong number of axes. Histogram axes : " 
                + ", ".join(histProvider.projectedHist.axisNames()))

        self.figure = bokeh.plotting.figure(**kwargs)

        self.histProvider = histProvider

        h_proj = self.histProvider.projectedHist

        self.figure.title = histProvider.hist.label
        self.figure.xaxis.axis_label = h_proj.axes[0].label
        self.figure.yaxis.axis_label = h_proj.axes[1].label

        colorMapper = bokeh.models.LinearColorMapper(palette="Spectral11")
        self.plottedValueTitle = bokeh.models.Title(text="")
        
        self.source = ColumnDataSource()
        self.update(None, None, None)
        self.figure.image(image='histogram_2D_view', x=h_proj.axes[0].edges[0], y=h_proj.axes[0].edges[0],
            dw=h_proj.axes[0].edges[-1]-h_proj.axes[0].edges[0], dh=h_proj.axes[0].edges[-1]-h_proj.axes[0].edges[0],
            color_mapper=colorMapper,
            source=self.source)

        self.figure.add_layout(bokeh.models.ColorBar(color_mapper=colorMapper), 'right')

        
        self.figure.add_layout(self.plottedValueTitle, "right")

        self.histProvider.registerUpdateCallback(self.update)
    
    def update(self, attr, old, new):
        # It would seem that the histogram x, y view is the transpose of what is expected by bokeh, though it needs to be checked
        self.source.data = {"histogram_2D_view":[np.transpose(self.histProvider.getProjectedHistogramView())]}

        if self.histProvider.isProfileEnabled():
            self.plottedValueTitle.text = self.histProvider.hist.profileOn.label
        else:
            self.plottedValueTitle.text = 'Event count'



