from functools import partial

from HistogramLib.projection_manager import HistogramProjectedView
from bokeh_apps.widgets import *

args = parseArgs()
histStore = HistogramStore(args.hist_folder)

""" z position of all layers (nb 1 to 28)"""
layers_z = [13.877500, 14.767500, 16.782499, 17.672501, 19.687500, 20.577499, 22.692499, 23.582500, 25.697500, 26.587500, 28.702499, 29.592501, 31.507500, 32.397499, 34.312500, 35.202499, 37.117500, 38.007500, 39.922501, 40.812500, 42.907501, 44.037498, 46.412498, 47.542500, 49.681999, 50.688000, 52.881500, 53.903500]

class Selectors:
    def __init__(self) -> None:
        self.datatype_selector = DatatypeSelector()
        self.clueParamSelector = ClueParamsSelector(histStore.getPossibleClueParameters())
        #self.clueParamSelector = PlaceholderClueParamsSelector()
        self.layerSelector = makeLayerSelector()
        self.beamEnergySelector = makeBeamEnergySelector()
        self.histKindSelector = HistogramKindRadioButton()
        self.mainOrAllTrackstersSelector = makeMainOrAllTrackstersSelector() 

        self.MakeView = partial(HistogramProjectedView, histStore, 
            shelfIdProviders=[self.datatype_selector, self.clueParamSelector],
            projectionProviders={'beamEnergy': self.beamEnergySelector, 'layer': self.layerSelector},
            histKindSelector=self.histKindSelector)

        self.MakeViewClue3D = partial(HistogramProjectedView, histStore, 
            shelfIdProviders=[self.datatype_selector, self.clueParamSelector],
            projectionProviders={'beamEnergy': self.beamEnergySelector, 'layer': self.layerSelector, 'mainOrAllTracksters': self.mainOrAllTrackstersSelector},
            histKindSelector=self.histKindSelector)
