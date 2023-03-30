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
                "RechitsTotalEnergyPerEvent", "EnergyClustered2DPerEvent", "Clus3DClusteredEnergy"])],
            singlePlotClass=None, multiPlotClass=StepHistogram1D,
            ).model
        ),
        TabPanel(title="Total energy (fraction)",
            child=PlotManager(store=histStore, 
            selectors=s.selectorsClue3D + [HistogramIdNameMultiSelector([
                "RechitsTotalEnergyFractionPerEvent", "FractionEnergyClustered2DPerEvent", "Clus3DClusteredFractionEnergy"])],
            singlePlotClass=None, multiPlotClass=StepHistogram1D,
            ).model
        ),
        TabPanel(title="Total energy (profile)",
            child=PlotManager(store=histStore, 
            selectors=s.selectorsClue3D + [HistogramIdNameMultiSelector([
                "RechitsMeanTotalEnergyPerEvent", "MeanEnergyClustered2DPerEvent", "Clus3DMeanClusteredEnergy"])],
            singlePlotClass=None, multiPlotClass=LineHistogram1D,
            ).model
        ),
        TabPanel(title="Energy per layer",
            child=PlotManager(store=histStore, 
            selectors=s.selectorsClue3D + [
                HistogramIdNameMultiSelector([
                    "RechitsEnergyReconstructedPerLayer", "EnergyClustered2DPerLayer", "Clus3DClusteredEnergyPerLayer"])
            ],
            singlePlotClass=None, multiPlotClass=LineHistogram1D
            ).model
        ),
        s.tabStandard("TrueBeamEnergy", "TrueBeamEnergy")
    ])
))

