from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnClue3D(),
    Tabs(tabs=[
        s.tabClue3D("Transverse p.(barycenter)", "Clus3DRechitsDistanceToBarycenter_EnergyFractionNormalized"),
        s.tabClue3D("Transverse p.(barycenter, area norm.)", "Clus3DRechitsDistanceToBarycenter_AreaNormalized", y_axis_type="log"),
        s.tabClue3D("Transverse p.(impact)", "Clus3DRechitsDistanceToImpact_EnergyFractionNormalized"),
        s.tabClue3D("Transverse p.(impact, area norm.)", "Clus3DRechitsDistanceToImpact_AreaNormalized", y_axis_type="log"),
    ])
))
