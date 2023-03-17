from bokeh.layouts import row
from bokeh.plotting import curdoc
from bokeh.models import Tabs, TabPanel

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnClue3D(),
    Tabs(tabs=[
        TabPanel(title="Total energy",
            child=PlotManager(store=histStore, 
            selectors=s.selectorsClue3D + [HistogramIdNameMultiSelector([
                "RechitsTotalEnergyClusteredPerEvent", "EnergyClustered2DPerEvent", "Clus3DClusteredEnergy"])],
            singlePlotClass=None, multiPlotClass=StepHistogram1D,
            ).model
        )
    ])
))

