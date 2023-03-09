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
        BokehHistogram(s.MakeView(histName="RechitsEnergy")).figure,
        MultiBokehHistogram2D(s.MakeView(histName="RechitsPositionXY")).figure,
        BokehHistogram(s.MakeView(histName="RechitsPositionZ"), xGridTicks=layers_z).figure,
    ],
    [ # Second line 
        BokehHistogram(s.MakeView(histName="RechitsRho")).figure,
        BokehHistogram(s.MakeView(histName="RechitsDelta")).figure,
        MultiBokehHistogram2D(s.MakeView(histName="RechitsRhoDelta")).figure,
    ]
    
]))
