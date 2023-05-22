import itertools
from functools import cached_property

import plotly.express as px
import plotly.graph_objects as go

from event_visualizer.event_index import LoadedEvent
from event_visualizer.plotter.utils import *


class LongitudinalProfileVisualization(BaseVisualization):
    def __init__(self, event:LoadedEvent) -> None:
        super().__init__(event)
        self.fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=(
                    f"Longitudinal shower profile - ntuple {event.record.ntupleNumber}, event {event.record.event} "
                    f"- e+ {event.record.beamEnergy} GeV"
                )),
                autosize=True,
                xaxis_title="Layer",
                yaxis_title="Rechit energy sum (GeV)",
                barmode="overlay",
            ), 
        )


    def addRechitsProfile(self):
        self.fig.add_trace(go.Histogram(
            x=self.event.record.rechits_layer,
            y=self.event.record.rechits_energy,
            # needed otherwise plotly starts to rebin things
            nbinsx=int(np.max(self.event.record.rechits_layer)-np.min(self.event.record.rechits_layer)+1), 
            name="Rechits energy sum",
            histfunc="sum", # needed !
            hovertemplate="Layer %{x}<br>Energy sum %{y} GeV"
        ))
        return self
    
    def addClueProfile(self):
        self.fig.add_trace(go.Histogram(
            x=self.event.clus2D_df.clus2D_layer,
            y=self.event.clus2D_df.clus2D_energy,
            # needed otherwise plotly starts to rebin things
            nbinsx=int(self.event.clus2D_df.clus2D_layer.max()-self.event.clus2D_df.clus2D_layer.min()+1), 
            name="CLUE energy sum",
            histfunc="sum", # needed !
            hovertemplate="Layer %{x}<br>Energy sum %{y} GeV"
        ))
        return self
    
    