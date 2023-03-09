from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(layout(
[
    [ # First line
        column(
            s.clueParamSelector.widget, s.datatype_selector.widget, s.beamEnergySelector.widget,
            s.layerSelector.widget, s.histKindSelector.widget
        ),
        BokehHistogram(s.MakeView(histName="EnergyClustered2DPerLayer")).figure,
        BokehHistogram(s.MakeView(histName="LayerWithMaximumClustered2DEnergy")).figure,
        BokehHistogram(s.MakeView(histName="NumberOf2DClustersPerLayer")).figure
    ],
    [ # Second line 
        BokehHistogram(s.MakeView(histName="Cluster2DRho")).figure,
        BokehHistogram(s.MakeView(histName="Cluster2DDelta")).figure,
        MultiBokehHistogram2D(s.MakeView(histName="Cluster2DRhoDelta")).figure,
    ]
    
]))
