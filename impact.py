from bokeh.layouts import layout
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *
from HistogramLib.bokeh.histogram_widget import *

s = Selectors()

curdoc().add_root(layout(
[
    [ # First line
        s.makeWidgetColumnStandard(),
        s.MakePlotStandard('ImpactXY', plotType="2d")
    ],
]))
