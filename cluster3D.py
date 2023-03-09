from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(layout(
[
    [ # First line
        column(
            s.clueParamSelector.widget, s.datatype_selector.widget, s.beamEnergySelector.widget,
            s.layerSelector.widget, s.histKindSelector.widget, s.mainOrAllTrackstersSelector.widget
        ),
        MultiBokehHistogram2D(s.MakeViewClue3D(histName="Clus3DSpatialResolution")).figure,
        MultiBokehHistogram2D(s.MakeViewClue3D(histName="Clus3DPositionXY")).figure,
        BokehHistogram(s.MakeViewClue3D(histName="Clus3DPositionZ")).figure
    ],
    [ # Second line 
        BokehHistogram(s.MakeViewClue3D(histName="Clus3DFirstLayerOfCluster")).figure,
        BokehHistogram(s.MakeViewClue3D(histName="Clus3DLastLayerOfCluster")).figure,
        BokehHistogram(s.MakeViewClue3D(histName="Clus3DNumberOf2DClustersPerLayer")).figure,
    ]
    
]))
