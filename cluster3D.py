from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnClue3D(),
    Tabs(tabs=[
        s.tabClue3D("Position Z", "Clus3DPositionZ"),
        s.tabClue3D("Total energy", "Clus3DClusteredEnergy"),
        s.tabClue3D("Nb of 2D clusters per layer", "Clus3DNumberOf2DClustersPerLayer"),
        s.tabClue3D("First layer of cluster", "Clus3DFirstLayerOfCluster"),
        s.tabClue3D("Last layer of cluster", "Clus3DLastLayerOfCluster"), 
        s.tabClue3D("Energy per layer", "Clus3DClusteredEnergyPerLayer")
    ])
))
