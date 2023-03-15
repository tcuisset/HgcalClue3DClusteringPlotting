from bokeh.layouts import layout
from bokeh.plotting import curdoc
from bokeh.models import Tabs, TabPanel

from bokeh_apps.common_init import *
from HistogramLib.bokeh.histogram_widget import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnStandard(),
    Tabs(tabs=[
        s.tabStandard("Energy", "RechitsEnergy"),
        s.tabStandard("Position X-Y", "RechitsPositionXY", plotType="2d"),
        s.tabStandard("Position layer", "RechitsPositionLayer"),
        s.tabStandard("Rho", "RechitsRho"),
        s.tabStandard("Delta", "RechitsDelta"),
        s.tabStandard("Rho-Delta", "RechitsRhoDelta", plotType="2d"),
    ])
))
