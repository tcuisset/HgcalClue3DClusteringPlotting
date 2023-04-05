from bokeh.layouts import layout
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *
from HistogramLib.bokeh.histogram_widget import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnStandard(),
    Tabs(tabs=[
        s.tabStandard("Impact X-Y", 'ImpactXY', plotType="2d")
    ])
))

curdoc().title = "Impact from DWC"
