from functools import partial

from HistogramLib.projection_manager import HistogramProjectedView
from bokeh_apps.widgets import *

args = parseArgs()
histStore = HistogramStore(args.hist_folder)


class Selectors:
    def __init__(self) -> None:
        self.datatype_selector = DatatypeSelector()
        self.clueParamSelector = ClueParamsSelector(histStore.getPossibleClueParameters())
        #self.clueParamSelector = PlaceholderClueParamsSelector()
        self.layerSelector = makeLayerSelector()
        self.beamEnergySelector = makeBeamEnergySelector()
        self.toggleProfile = ToggleProfileButton()
        self.mainOrAllTrackstersSelector = makeMainOrAllTrackstersSelector() 

        self.MakeView = partial(HistogramProjectedView, histStore, 
            shelfIdProviders=[self.datatype_selector, self.clueParamSelector],
            projectionProviders={'beamEnergy': self.beamEnergySelector, 'layer': self.layerSelector},
            toggleProfileButton=self.toggleProfile)

        self.MakeViewClue3D = partial(HistogramProjectedView, histStore, 
            shelfIdProviders=[self.datatype_selector, self.clueParamSelector],
            projectionProviders={'beamEnergy': self.beamEnergySelector, 'layer': self.layerSelector, 'mainOrAllTracksters': self.mainOrAllTrackstersSelector},
            toggleProfileButton=self.toggleProfile)
