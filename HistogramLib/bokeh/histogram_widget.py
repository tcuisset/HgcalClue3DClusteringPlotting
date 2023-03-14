from typing import Dict
import itertools
import bokeh.models
import bokeh.plotting
import bokeh.palettes
import boost_histogram as bh
import numpy as np
from bokeh.models import ColumnDataSource, FixedTicker


from ..histogram import *
from ..projection_manager import *
from ..store import *
from ..selectors import *
from ..plot_manager import *


class AbstractHistogram(AbstractPlotClass):
    plottedDimensionsCount:int = 1 # How many dimensions are plotted (1 or 2)
    def __init__(self, metadata:HistogramMetadata, xGridTicks=None, **kwargs_figure) -> None:
        """ xGridTicks : Put x grid lines at these x values """
        if len(metadata.axes) != self.plottedDimensionsCount:
            raise ValueError(f"You are trying to plot a {self.plottedDimensionsCount}D histogram using data with {len(metadata.axes)} dimensions, you may be missing a projection. Histogram axes : " 
                + ", ".join([axis.name for axis in metadata.axes]))

        self.metadata = metadata
        self.figure = bokeh.plotting.figure(**kwargs_figure)

        
        self.figure.title = self.metadata.title
        self.figure.xaxis.axis_label = self.metadata.axes[0].label
        if self.plottedDimensionsCount == 2:
            self.figure.yaxis.axis_label = self.metadata.axes[1].label
        
        if xGridTicks:
            self.figure.xgrid.ticker = FixedTicker(ticks=xGridTicks)

        self.source = ColumnDataSource()
    

class AbstractSingleHistogram(AbstractHistogram):
    def __init__(self, *args, **kwargs) -> None:
        self.histProjectedView:HistogramView = kwargs.pop("projectedView")
        super().__init__(*args, **kwargs)
    
    def update(self) -> None:
        self.histProjectedView.update()

class QuadHistogram1D(AbstractSingleHistogram):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.update()
        self.figure.quad(bottom=0, source=self.source)
    
    def update(self):
        super().update()
        try:
            self.source.data = {"top":self.histProjectedView.getProjectedHistogramView(), "left":self.metadata.axes[0].edges[:-1], "right":self.metadata.axes[0].edges[1:]}
        except HistogramLoadError:
            self.source.data = {}
        self.figure.yaxis.axis_label = self.metadata.getPlotLabel(self.histProjectedView.histKind)


class QuadHistogram2D(AbstractSingleHistogram):
    plottedDimensionsCount = 2
    def __init__(self,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        colorMapper = bokeh.models.LinearColorMapper(palette="Spectral11")
        self.plottedValueTitle = bokeh.models.Title(text="")
        
        self.update()
        axes = self.metadata.axes
        self.figure.image(image='histogram_2D_view', x=axes[0].edges[0], y=axes[1].edges[0],
            dw=axes[0].edges[-1]-axes[0].edges[0], dh=axes[1].edges[-1]-axes[1].edges[0],
            color_mapper=colorMapper,
            source=self.source)

        self.figure.add_layout(bokeh.models.ColorBar(color_mapper=colorMapper), 'right')
        self.figure.add_layout(self.plottedValueTitle, "right")
    
    def update(self):
        super().update()
        try:
            # It would seem that the histogram x, y view is the transpose of what is expected by bokeh, though it needs to be checked
            self.source.data = {"histogram_2D_view":[np.transpose(self.histProjectedView.getProjectedHistogramView())]}
        except HistogramLoadError:
            self.source.data = {}
        self.figure.yaxis.axis_label = self.metadata.getPlotLabel(self.histProjectedView.histKind)


class AbstractMultiHistogram(AbstractHistogram):
    def __init__(self, projectedViews:Dict[str, HistogramView], *args, **kwargs) -> None:
        """ 
        Multiple overlayed histograms, one per provided HistogramProjectedView
        histProviders : dict legendName:str -> HistogramView 
        """
        self.histProjectedViews = projectedViews
        super().__init__(*args, **kwargs)

    def _setupLegend(self):
        """ To be called after calling glyph method so the legend exists"""
        self.figure.legend.location = "top_left"
        self.figure.legend.click_policy = "hide" # So we can click on legend to make line disappear

    @property
    def firstProvider(self) -> HistogramView:
        return next(iter(self.histProjectedViews.values()))
    
    def update(self) -> None:
        for histProjectedView in self.histProjectedViews.values():
            histProjectedView.update()

class LineHistogram1D(AbstractMultiHistogram):
    def __init__(self, *args, **kwargs) -> None:
        """ 
        Line plots with multiple lines, one per provided HistogramProjectedView
        histProviders : dict legendName:str -> HistogramProjectedView 
        """
        palette = kwargs.pop('palette', bokeh.palettes.Category10[10])
        super().__init__(*args, **kwargs)

        xAxis = self.metadata.axes[0]
        if issubclass(type(xAxis), bh.axis.IntCategory): #or issubclass(type(xAxis), bh.axis.StrCategory)
            # int category axis : we need to map bin indices to their values
            # using xAxis.size() does not include overflow/undeflow
            self.source.data["x_bins_edges"] = [xAxis.bin(index) for index in range(xAxis.size)]
        else:
            # continuous axis : just use bin edges
            self.source.data["x_bins_edges"] = xAxis.edges[:-1]
        self.update()

        colors = itertools.cycle(palette)
        for legend_name, color in zip(self.histProjectedViews.keys(), colors): 
            self.figure.line(source=self.source, x="x_bins_edges", y=legend_name,
                legend_label=legend_name, color=color)
            self.figure.circle(source=self.source, x="x_bins_edges", y=legend_name, legend_label=legend_name,
                fill_color="white", size=8, color=color)
        self._setupLegend()

    def update(self):
        super().update()
        for legend_name, histProvider in self.histProjectedViews.items():
            try:
                self.source.data[legend_name] = histProvider.getProjectedHistogramView()
            except HistogramLoadError:
                self.source.remove(legend_name)

        self.figure.yaxis.axis_label = self.metadata.getPlotLabel(self.firstProvider.histKind)

class StepHistogram1D(AbstractMultiHistogram):
    def __init__(self, *args, **kwargs) -> None:
        """ 
        Step plots (histograms) with multiple lines, one per provided HistogramProjectedView
        histProviders : dict legendName:str -> HistogramProjectedView 
        """ 
        palette = kwargs.pop('palette', bokeh.palettes.Category10[10])
        super().__init__(*args, **kwargs)

        xAxis = self.metadata.axes[0]
        if issubclass(type(xAxis), bh.axis.IntCategory): #or issubclass(type(xAxis), bh.axis.StrCategory)
            # int category axis : we need to map bin indices to their values
            # using xAxis.size() does not include overflow/undeflow
            raise ValueError("Category axes not supported yet")
            self.source.data["x_bins_left"] = [xAxis.bin(index) for index in range(xAxis.size)]
        else:
            # continuous axis : just use bin edges
            x_bins = np.insert(xAxis.edges, 0, xAxis.edges[0])# We duplicate the first bin edge so we can draw a vertical line extending to y=0
            self.source.data["x_bins_left"] = x_bins
        self.update()

        colors = itertools.cycle(palette)
        for legend_name, color in zip(self.histProjectedViews.keys(), colors): 
            self.figure.step(source=self.source, x="x_bins_left", y=legend_name, legend_label=legend_name, color=color)
        self._setupLegend()

    def update(self):
        super().update()
        for legend_name, histProvider in self.histProjectedViews.items():
            try:
                #Insert 0 at beginning and end so that we have vertical lines extending to y=0 at the beginning and end of histogram
                copied_view = np.insert(histProvider.getProjectedHistogramView(), 0, 0.)
                copied_view = np.append(copied_view, 0.)
                self.source.data[legend_name] = copied_view
            except HistogramLoadError:
                self.source.remove(legend_name)

        self.figure.yaxis.axis_label = self.metadata.getPlotLabel(self.firstProvider.histKind)

