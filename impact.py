from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(layout(
[
    [ # First line
        column(
            s.clueParamSelector.widget, s.datatype_selector.widget, s.beamEnergySelector.widget,
            
        ),
        MultiBokehHistogram2D(s.MakeView(histName="ImpactXY")).figure,
    ],
]))
