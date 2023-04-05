from bokeh.layouts import row
from bokeh.plotting import curdoc
from bokeh.models import Tabs

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnStandard(),
    Tabs(tabs=[
        s.tabStandard("Position XY", "Clus2DPositionXY", plotType="2d"),
        s.tabStandard("Rho-Delta", "Cluster2DRhoDelta", plotType="2d"),
        s.tabStandard("PointType", "Cluster2DPointType", plotType="2d")
    ])
))
curdoc().title = "2D clusters (2D plots)"
