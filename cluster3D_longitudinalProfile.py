from bokeh.layouts import layout, column
from bokeh.plotting import curdoc

from bokeh_apps.common_init import *

s = Selectors()

curdoc().add_root(row(
    s.makeWidgetColumnClue3DProfile(),
    Tabs(tabs=[
        s.tabClue3DProfile("Interval holding fraction (minLayer)", "Clus3DIntervalHoldingFractionOfEnergy_Min"),
        s.tabClue3DProfile("Interval holding fraction\n<br/>(maxLayer)", "Clus3DIntervalHoldingFractionOfEnergy_Max"),
        s.tabClue3DProfile("Interval holding fraction (length)", "Clus3DIntervalHoldingFractionOfEnergy_IntervalLength"),
    ])
))
