from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

selectorsWithoutBeamEnergy = s.selectorsClue3DLongitudinalProfile.copy()
selectorsWithoutBeamEnergy.remove(s.beamEnergySelector)

selectorsWithoutEnergyFraction = s.selectorsClue3DLongitudinalProfile.copy()
selectorsWithoutEnergyFraction.remove(s.intervalEnergyFractionSelector)

curdoc().add_root(row(
    s.makeWidgetColumnClue3DProfile(),
    Tabs(tabs=[
        s.tabClue3DProfile("Interval holding fraction (minLayer)", "Clus3DIntervalHoldingFractionOfEnergy_Min"),
        s.tabClue3DProfile("Interval holding fraction (maxLayer)", "Clus3DIntervalHoldingFractionOfEnergy_Max"),
        s.tabClue3DProfile("Interval holding fraction (length)", "Clus3DIntervalHoldingFractionOfEnergy_IntervalLength"),
        TabPanel(title="Interval holding fraction (length, fct beamEnergy)", 
            child=s.MakePlot("Clus3DIntervalHoldingFractionOfEnergy_MeanIntervalLength", selectors=selectorsWithoutBeamEnergy,
                plotType=LineHistogram1D)),
        TabPanel(title="Interval holding fraction (length, fct fraction)", 
            child=s.MakePlot("Clus3DIntervalHoldingFractionOfEnergy_MeanIntervalLength", selectors=selectorsWithoutEnergyFraction,
                plotType=LineHistogram1D)),
    ])
))
curdoc().title = "Longitudinal profile of 3D clusters"
