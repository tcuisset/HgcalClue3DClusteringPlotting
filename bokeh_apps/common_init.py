from functools import partial
from bokeh.layouts import column
from bokeh.models import Tabs, TabPanel

from HistogramLib.store import HistogramStore
from HistogramLib.plot_manager import PlotManager
from HistogramLib.bokeh.histogram_widget import *
from hists.store import HistogramId
from bokeh_apps.widgets import *


import argparse

def parseArgs():
    parser = argparse.ArgumentParser(description="Plotting code to be run using Bokeh server, use bokeh serve SCRIPT.py --args ARGS")
    parser.add_argument("--hist-folder", dest="hist_folder",
        help="path to folder holding all histograms. Will load recursively all clueparams and datatypes inside this folder")
    parser.add_argument("--single-file", dest='single_file',
        help="Only load a single pickle file (for debugging), given by this full path to the file")
    return parser.parse_args()

args = parseArgs()
histStore = HistogramStore(args.hist_folder, HistogramId)

""" z position of all layers (nb 1 to 28)"""
layers_z = [13.877500, 14.767500, 16.782499, 17.672501, 19.687500, 20.577499, 22.692499, 23.582500, 25.697500, 26.587500, 28.702499, 29.592501, 31.507500, 32.397499, 34.312500, 35.202499, 37.117500, 38.007500, 39.922501, 40.812500, 42.907501, 44.037498, 46.412498, 47.542500, 49.681999, 50.688000, 52.881500, 53.903500]

class Selectors:
    def __init__(self) -> None:
        clueParams = histStore.getPossibleClueParameters()
        if not clueParams:
            # Could not find any histograms
            raise RuntimeError("Could not find any saved histograms. Aborting.")
        self.clueParamSelector = ClueParamsSelector(clueParams)
        self.datatype_selector = makeDatatypeSelector(histStore.getPossibleDatatypes())
        #self.clueParamSelector = PlaceholderClueParamsSelector()
        self.layerSelector = makeLayerSelector()
        self.beamEnergySelector = makeBeamEnergySelector()
        self.pointTypeSelector = makePointTypeSelector()
        self.rechitEnergySelector = makeRechitEnergySelector()
        self.clus3DSizeSelector = makeCluster3DSizeSelector()
        self.histKindSelector = HistogramKindRadioButton()
        self.normalizePlots = DensityHistogramToggle()
        self.mainOrAllTrackstersSelector = makeMainOrAllTrackstersSelector()

        self.selectorsStandardBegin = [self.datatype_selector, self.clueParamSelector, 
                self.beamEnergySelector, self.layerSelector, self.pointTypeSelector]
        self.selectorsStandardEnd = [self.histKindSelector, self.normalizePlots]
        self.selectorsStandard = self.selectorsStandardBegin + self.selectorsStandardEnd

        self.selectorsRechitsBegin = self.selectorsStandardBegin + [self.rechitEnergySelector]
        self.selectorsRechits = self.selectorsRechitsBegin + self.selectorsStandardEnd

        self.selectorsClue3DBegin = self.selectorsStandardBegin + [self.mainOrAllTrackstersSelector,
            self.clus3DSizeSelector]
        self.selectorsClue3D = self.selectorsClue3DBegin + self.selectorsStandardEnd

    def MakePlot(self, histName:str, selectors, plotType:str|AbstractHistogram="1d", **kwargs):
        if plotType == "1d":
            singlePlotClass=QuadHistogram1D
            multiPlotClass=StepHistogram1D
        elif plotType == "2d":
            singlePlotClass=QuadHistogram2D
            multiPlotClass=None
        else:
            singlePlotClass = plotType
            multiPlotClass = plotType
        return PlotManager(store=histStore, 
            selectors=selectors + [HistogramIdNameSelector(histName)],
            singlePlotClass=singlePlotClass, multiPlotClass=multiPlotClass,
            **kwargs).model

    def tabStandard(self, tabTitle:str, *args, **kwargs):
        return TabPanel(title=tabTitle, child=self.MakePlot(*args, selectors=self.selectorsStandard, **kwargs))
    def tabRechits(self, tabTitle:str, *args, **kwargs):
        return TabPanel(title=tabTitle, child=self.MakePlot(*args, selectors=self.selectorsRechits, **kwargs))
    def tabClue3D(self, tabTitle:str, *args, **kwargs):
        return TabPanel(title=tabTitle, child=self.MakePlot(*args, selectors=self.selectorsClue3D, **kwargs))

    def makeWidgetColumnStandard(self):
        return column(*(selector.model for selector in self.selectorsStandard))
    def makeWidgetColumnRechits(self):
        return column(*(selector.model for selector in self.selectorsRechits))
    def makeWidgetColumnClue3D(self):
        return column(*(selector.model for selector in self.selectorsClue3D))