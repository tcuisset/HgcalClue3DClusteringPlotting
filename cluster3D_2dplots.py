from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnClue3D(),
    Tabs(tabs=[
        s.tabClue3D("Spatial resolution", "Clus3DSpatialResolutionOf2DClusters", plotType="2d"),
        s.tabClue3D("Position X-Y", "Clus3DPositionXY",  plotType="2d"),
    ])
))
