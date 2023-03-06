from functools import partial

from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from HistogramLib.histogram import *
from HistogramLib.bokeh import *
from bokeh_widgets import *

args = parseArgs()
histStore = HistogramStore(args.hist_folder)

datatype_selector = DatatypeSelector()
clueParamSelector = ClueParamsSelector(histStore.getPossibleClueParameters())
#clueParamSelector = PlaceholderClueParamsSelector()
layerSelector = makeLayerSelector()
beamEnergySelector = makeBeamEnergySelector()
toggleProfile = ToggleProfileButton()

MakeView = partial(HistogramProjectedView, histStore, [datatype_selector, clueParamSelector],
    {'beamEnergy' : beamEnergySelector, 'layer'  : layerSelector}, toggleProfileButton=toggleProfile)



curdoc().add_root(layout([
    [column(
        clueParamSelector.widget,
        datatype_selector.widget, beamEnergySelector.widget, layerSelector.widget, toggleProfile.widget),
    MultiBokehHistogram2D(MakeView(histName="Clus3DSpatialResolution")).figure],
    #[h_xy.figure, h_spatial_resolution_xy.figure, h_spatial_resolution_xy_profiled.figure]
]))