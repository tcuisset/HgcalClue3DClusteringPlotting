from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnClue3DShowerPosition(),
    Tabs(tabs=[
        s.tabClue3D("Distance DWC impact/barycenter", "Clus3DImpactVsBarycenter"),
        s.tabClue3DShowerPosition("Angle PCA/Impact", "Clus3DAnglePCAToImpact"),
        s.tabClue3DShowerPosition("Angle PCA (cleaned)/Impact (Oxz)", "Clus3DAnglePCAToImpact_XY", forcePlotAxises=["clus3D_angle_pca_impact_x"]),
        s.tabClue3DShowerPosition("Angle PCA (cleaned)/Impact (Oyz)", "Clus3DAnglePCAToImpact_XY", forcePlotAxises=["clus3D_angle_pca_impact_y"]),
        s.tabClue3DShowerPosition("Angle PCA (cleaned)/Impact (X-Y 2D)", "Clus3DAnglePCAToImpact_XY", plotType="2d"),
    ])
))

curdoc().title = "3D clusters (shower position)"
