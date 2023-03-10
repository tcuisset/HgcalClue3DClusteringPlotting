from typing import Dict
import itertools
import bokeh.models
import bokeh.plotting
import bokeh.palettes
import boost_histogram as bh
import numpy as np
from bokeh.models import ColumnDataSource, FixedTicker

from .histogram import *
from .projection_manager import *
from .store import *


class RangeAxisSelector(ProjectionAxisSelector):
    def __init__(self, axisName:str, **kwargs) -> None:
        self.axisName = axisName
        self.widget = bokeh.models.RangeSlider(**kwargs)

    def getSlice(self):
        (min, max) = self.widget.value
        return RangeHistogramSlice(self.axisName, int(min), int(max))
    
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

class RadioButtonGroupAxisSelector(ProjectionAxisSelector):
    """ You need to pass labels as kwargs with the same labels as category axis values """
    def __init__(self, axisName:str, **kwargs) -> None:
        self.axisName = axisName
        self.widget = bokeh.models.RadioButtonGroup(**kwargs)
    
    def getSlice(self):
        return SingleValueHistogramSlice(self.axisName, self.widget.labels[self.widget.active])
    def registerCallback(self, callback):
        self.widget.on_change('active', callback)

class SliderMinWithOverflowAxisSelector(ProjectionAxisSelector):
    """ Projects on [value:] with upper overflow bin included (undeflow excluded)"""
    def __init__(self, axisName:str, **kwargs) -> None:
        self.axisName = axisName
        self.widget = bokeh.models.Slider(**kwargs)

    def getSlice(self):
        min = self.widget.value
        return MinWithOverflowHistogramSlice(self.axisName, int(min))
    
    def registerCallback(self, callback):
        #Use value_throttled so that it updates only on mouse release to avoid recomputing all the time when dragging
        self.widget.on_change('value_throttled', callback)


class BokehHistogram:
    def __init__(self, histProvider:HistogramProjectedView, xGridTicks=None, **kwargs_figure) -> None:
        """ xGridTicks : Put x grid lines at these x values """
        if len(histProvider.projectedHist.axes) != 1:
            raise ValueError("You are trying to plot a 1D histogram using multidimensional data, you are missing a projection. Histogram axes : " 
                + ", ".join([axis.name for axis in histProvider.projectedHist]))

        self.figure = bokeh.plotting.figure(**kwargs_figure)
        self.histProvider = histProvider
        
        self.figure.title = histProvider.myHist.label
        self.figure.xaxis.axis_label = self.histProvider.projectedHist.axes[0].label
        if xGridTicks:
            self.figure.xgrid.ticker = FixedTicker(ticks=xGridTicks)

        self.source = ColumnDataSource()
        self.update()
        self.figure.quad(bottom=0, source=self.source)

        self.histProvider.registerUpdateCallback(self.update)
    
    def update(self):
        h_proj = self.histProvider.projectedHist
        self.source.data = {"top":self.histProvider.getProjectedHistogramView(), "left":h_proj.axes[0].edges[:-1], "right":h_proj.axes[0].edges[1:]}

        self.figure.yaxis.axis_label = self.histProvider.getHistogramBinCountLabel()


class MultiBokehHistogram2D:
    def __init__(self, histProvider:HistogramProjectedView, **kwargs) -> None:
        if len(histProvider.projectedHist.axes) != 2:
            raise ValueError("You are trying to plot a 2D histogram with the wrong number of axes. Histogram axes : " 
                + ", ".join([axis.name for axis in histProvider.projectedHist]))

        self.figure = bokeh.plotting.figure(**kwargs)

        self.histProvider = histProvider

        h_proj = self.histProvider.projectedHist

        self.figure.title = histProvider.myHist.label
        self.figure.xaxis.axis_label = h_proj.axes[0].label
        self.figure.yaxis.axis_label = h_proj.axes[1].label

        colorMapper = bokeh.models.LinearColorMapper(palette="Spectral11")
        self.plottedValueTitle = bokeh.models.Title(text="")
        
        self.source = ColumnDataSource()
        self.update()
        self.figure.image(image='histogram_2D_view', x=h_proj.axes[0].edges[0], y=h_proj.axes[0].edges[0],
            dw=h_proj.axes[0].edges[-1]-h_proj.axes[0].edges[0], dh=h_proj.axes[0].edges[-1]-h_proj.axes[0].edges[0],
            color_mapper=colorMapper,
            source=self.source)

        self.figure.add_layout(bokeh.models.ColorBar(color_mapper=colorMapper), 'right')

        
        self.figure.add_layout(self.plottedValueTitle, "right")

        self.histProvider.registerUpdateCallback(self.update)
    
    def update(self):
        # It would seem that the histogram x, y view is the transpose of what is expected by bokeh, though it needs to be checked
        self.source.data = {"histogram_2D_view":[np.transpose(self.histProvider.getProjectedHistogramView())]}

        self.plottedValueTitle.text = self.histProvider.getHistogramBinCountLabel()


class BokehMultiLine:
    def __init__(self, histProviders:Dict[str, HistogramProjectedView], **kwargs_figure) -> None:
        """ 
        Line plots with multiple lines, one per provided HistogramProjectedView
        histProviders : dict legendName:str -> HistogramProjectedView 
        """
        self.histProviders = histProviders

        if len(self.firstProvider.projectedHist.axes) != 1:
            raise ValueError("You are trying to plot a 1D histogram using multidimensional data, you are missing a projection. Histogram axes : " 
                + ", ".join([axis.name for axis in self.firstProvider.projectedHist.axes]))

        self.figure = bokeh.plotting.figure(**kwargs_figure)
        
        
        self.figure.title = self.firstProvider.myHist.label
        self.figure.xaxis.axis_label = self.firstProvider.projectedHist.axes[0].label


        self.source = ColumnDataSource()
        projHist = self.firstProvider.projectedHist
        xAxis = projHist.axes[0]
        if issubclass(type(xAxis), bh.axis.IntCategory): #or issubclass(type(xAxis), bh.axis.StrCategory)
            # int category axis : we need to map bin indices to their values
            # using xAxis.size() does not include overflow/undeflow
            self.source.data["x_bins_edges"] = [xAxis.bin(index) for index in range(xAxis.size)]
        else:
            # continuous axis : just use bin edges
            self.source.data["x_bins_edges"] = projHist.axes[0].edges[:-1]
        self.update()

        colors = itertools.cycle(bokeh.palettes.Category10[10])
        for legend_name, color in zip(self.histProviders.keys(), colors): 
            self.figure.line(source=self.source, x="x_bins_edges", y=legend_name,
                legend_label=legend_name, color=color)
            self.figure.circle(source=self.source, x="x_bins_edges", y=legend_name, legend_label=legend_name,
                fill_color="white", size=8, color=color)

        self.figure.legend.location = "top_left"
        self.figure.legend.click_policy = "hide" # So we can click on legend to make line disappear

        # We need to register on all histProvider as otherwise self.update will be called by the first histProvider to be updated
        # whilst the other ones will still have the old projections 
        # So self.update will be called lots of time, leading to mixed histograms until the last call where all histProvider have updated
        # TODO this is very inefficient
        for histProvider in self.histProviders.values():
            histProvider.registerUpdateCallback(self.update)
        
    
    @property
    def firstProvider(self) -> HistogramProjectedView:
        return next(iter(self.histProviders.values()))

    def update(self):
        for legend_name, histProvider in self.histProviders.items():
            self.source.data[legend_name] = histProvider.getProjectedHistogramView()

        self.figure.yaxis.axis_label = self.firstProvider.getHistogramBinCountLabel()

class BokehMultiStep:
    def __init__(self, histProviders:Dict[str, HistogramProjectedView], **kwargs_figure) -> None:
        """ 
        Step plots (histograms) with multiple lines, one per provided HistogramProjectedView
        histProviders : dict legendName:str -> HistogramProjectedView 
        """
        self.histProviders = histProviders

        if len(self.firstProvider.projectedHist.axes) != 1:
            raise ValueError("You are trying to plot a 1D histogram using multidimensional data, you are missing a projection. Histogram axes : " 
                + ", ".join([axis.name for axis in self.firstProvider.projectedHist.axes]))

        self.figure = bokeh.plotting.figure(**kwargs_figure)
        
        
        self.figure.title = self.firstProvider.myHist.label
        self.figure.xaxis.axis_label = self.firstProvider.projectedHist.axes[0].label


        self.source = ColumnDataSource()
        projHist = self.firstProvider.projectedHist
        xAxis = projHist.axes[0]
        if issubclass(type(xAxis), bh.axis.IntCategory): #or issubclass(type(xAxis), bh.axis.StrCategory)
            # int category axis : we need to map bin indices to their values
            # using xAxis.size() does not include overflow/undeflow
            raise ValueError("Category axes not supported yet")
            self.source.data["x_bins_left"] = [xAxis.bin(index) for index in range(xAxis.size)]
        else:
            # continuous axis : just use bin edges
            x_bins = np.insert(projHist.axes[0].edges, 0, projHist.axes[0].edges[0])# We duplicate the first bin edge so we can draw a vertical line extending to y=0
            self.source.data["x_bins_left"] = x_bins
        self.update()

        colors = itertools.cycle(bokeh.palettes.Category10[10])
        for legend_name, color in zip(self.histProviders.keys(), colors): 
            self.figure.step(source=self.source, x="x_bins_left", y=legend_name, legend_label=legend_name, color=color)
            

        self.figure.legend.location = "top_left"
        self.figure.legend.click_policy = "hide" # So we can click on legend to make line disappear

        # We need to register on all histProvider as otherwise self.update will be called by the first histProvider to be updated
        # whilst the other ones will still have the old projections 
        # So self.update will be called lots of time, leading to mixed histograms until the last call where all histProvider have updated
        # TODO this is very inefficient
        for histProvider in self.histProviders.values():
            histProvider.registerUpdateCallback(self.update)
        
    
    @property
    def firstProvider(self) -> HistogramProjectedView:
        return next(iter(self.histProviders.values()))

    def update(self):
        for legend_name, histProvider in self.histProviders.items():
            #Insert 0 at beginning and end so that we have vertical lines extending to y=0 at the beginning and end of histogram
            copied_view = np.insert(histProvider.getProjectedHistogramView(), 0, 0.)
            copied_view = np.append(copied_view, 0.)
            self.source.data[legend_name] = copied_view

        self.figure.yaxis.axis_label = self.firstProvider.getHistogramBinCountLabel()