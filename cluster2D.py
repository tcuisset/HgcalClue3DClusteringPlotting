from bokeh.layouts import row
from bokeh.plotting import curdoc
from bokeh.models import Tabs

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnStandard(),
    Tabs(tabs=[
        s.tabStandard("Position XY", "Clus2DPositionXY", plotType="2d"),
        s.tabStandard("Position layer", "Clus2DPositionLayer"),
        s.tabStandard("Energy per layer", "EnergyClustered2DPerLayer"),
        s.tabStandard("Layer with max energy", "LayerWithMaximumClustered2DEnergy"),
        s.tabStandard("Nb of clusters per layer", "NumberOf2DClustersPerLayer"),
        s.tabStandard("Rho", "Cluster2DRho"),
        s.tabStandard("Delta", "Cluster2DDelta"),
        s.tabStandard("Rho-Delta", "Cluster2DRhoDelta", plotType="2d"),
    ])
))

