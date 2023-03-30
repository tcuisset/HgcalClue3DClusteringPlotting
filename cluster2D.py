from bokeh.layouts import row
from bokeh.plotting import curdoc
from bokeh.models import Tabs

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnStandard(),
    Tabs(tabs=[
        s.tabStandard("Position layer", "Clus2DPositionLayer"),
        s.tabStandard("Total energy", "EnergyClustered2DPerEvent"),
        s.tabStandard("Total energy (fraction)", "FractionEnergyClustered2DPerEvent"),
        s.tabStandard("Total energy (profile)", "MeanEnergyClustered2DPerEvent", plotType=LineHistogram1D),
        s.tabStandard("Energy per layer", "EnergyClustered2DPerLayer"),
        s.tabStandard("Layer with max energy", "LayerWithMaximumClustered2DEnergy"),
        s.tabStandard("Layer with max energy (profile)", "MeanLayerWithMaximumClustered2DEnergy", plotType=LineHistogram1D),
        s.tabStandard("Cluster size", "Clus2DSize"),
        s.tabStandard("Nb of clusters per layer", "NumberOf2DClustersPerLayer"),
        s.tabStandard("Rho", "Cluster2DRho"),
        s.tabStandard("Delta", "Cluster2DDelta"),
    ])
))

