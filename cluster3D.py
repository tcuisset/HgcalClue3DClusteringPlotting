from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnClue3D(),
    Tabs(tabs=[
        s.tabClue3D("Position Z", "Clus3DPositionZ"),
        s.tabClue3D("Position Z (profile)", "Clus3DMeanPositionZFctBeamEnergy", plotType=LineHistogram1D),
        s.tabClue3D("Total energy", "Clus3DClusteredEnergy"),
        s.tabClue3D("Total energy (fraction)", "Clus3DClusteredFractionEnergy"),
        s.tabClue3D("Total energy (profile)", "Clus3DMeanClusteredEnergy", plotType=LineHistogram1D),
        s.tabClue3D("Cluster size", "Clus3DSize"),
        s.tabClue3D("Nb of 2D clusters per layer", "Clus3DNumberOf2DClustersPerLayer"),
        s.tabClue3D("First layer of cluster", "Clus3DFirstLayerOfCluster"),
        s.tabClue3D("Last layer of cluster", "Clus3DLastLayerOfCluster"), 
        s.tabClue3D("Energy per layer", "Clus3DClusteredEnergyPerLayer"),
        s.tabClue3D("Layer with max energy", "Clus3DLayerWithMaximumClusteredEnergy"),
        s.tabClue3D("Layer with max energy (profile)", "Clus3DMeanLayerWithMaximumClusteredEnergy", plotType=LineHistogram1D),
        s.tabClue3D("Distance DWC impact/barycenter", "Clus3DImpactVsBarycenter"),
    ])
))
