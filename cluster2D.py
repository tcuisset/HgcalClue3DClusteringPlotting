from bokeh.layouts import layout, column, row
from bokeh.plotting import curdoc
from bokeh.models import Tabs, TabPanel

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    column(
        s.clueParamSelector.widget, s.datatype_selector.widget, s.beamEnergySelector.widget,
        s.layerSelector.widget, s.histKindSelector.widget
    ),
    Tabs(tabs=[
        TabPanel(title="2D Cluster position", child=row(
            MultiBokehHistogram2D(s.MakeView(histName="Clus2DPositionXY")).figure,
            BokehHistogram(s.MakeView(histName="Clus2DPositionZ")).figure,
        )),
        TabPanel(title="Layer information", child=row(
            BokehHistogram(s.MakeView(histName="EnergyClustered2DPerLayer")).figure,
            BokehHistogram(s.MakeView(histName="LayerWithMaximumClustered2DEnergy")).figure,
            BokehHistogram(s.MakeView(histName="NumberOf2DClustersPerLayer")).figure,
        )),
        TabPanel(title="CLUE3D variables", child=row(
            BokehHistogram(s.MakeView(histName="Cluster2DRho")).figure,
        BokehHistogram(s.MakeView(histName="Cluster2DDelta")).figure,
        MultiBokehHistogram2D(s.MakeView(histName="Cluster2DRhoDelta")).figure,
        )),
    ])
))

        # BokehMultiStep({
        #     str(beamEnergy) : HistogramProjectedView(histStore, histName="EnergyClustered2DPerLayer",
        #         shelfIdProviders=[s.datatype_selector, s.clueParamSelector],
        #         projectionProviders={"beamEnergy" : PlaceholderAxisSelector(SingleValueHistogramSlice("beamEnergy", beamEnergy))},
        #         histKindSelector=FixedHistKindSelector(HistogramKind.PROFILE))
        #     for beamEnergy in beamEnergies
        # }).figure
