from bokeh.layouts import layout
from bokeh.plotting import curdoc
from bokeh.models import Tabs, TabPanel

from bokeh_apps.common_init import *
from HistogramLib.bokeh.histogram_widget import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnRechits(),
    Tabs(tabs=[
        s.tabRechits("Energy", "RechitsEnergy"),
        s.tabRechits("Energy (logY)", "RechitsEnergy", y_axis_type="log"),
        s.tabRechits("Total energy per event", "RechitsTotalEnergyClusteredPerEvent"),
        s.tabRechits("Position layer", "RechitsPositionLayer"),
        s.tabRechits("Rho", "RechitsRho"),
        s.tabRechits("Delta", "RechitsDelta"),
        s.tabRechits("PointType (1D)", "RechitsPointType"),
    ])
))

