import itertools
from functools import cached_property

import plotly.express as px
import plotly.graph_objects as go

from hists.dataframe import *
from utils import *


class Clue3DVisualization(BaseVisualization):
    def __init__(self, comp:DataframeComputations, eventNb) -> None:
        super().__init__(comp, eventNb)
        self.fig = create3DFigure()

        color_cycle = itertools.cycle(px.colors.qualitative.Plotly)
        self.mapClus3Did_color = {clus3D_id : next(color_cycle) for clus3D_id in self.clus3D_df.index.get_level_values(0).drop_duplicates().to_list()}
        self.mapClus2Did_color = {clus2D_id : next(color_cycle) for clus2D_id in self.clus2D_df.index.get_level_values("clus2D_id").drop_duplicates().to_list()}

    def makeAll(self) -> go.Figure:
        self.add3DClusters()
        self.add2DClusters()
        return self.fig




    def add3DClusters(self):
        self.fig.add_trace(go.Scatter3d(
            name="3D tracksters",
            x=self.clus3D_df["clus3D_x"], y=self.clus3D_df["clus3D_y"], z=self.clus3D_df["clus3D_z"], 
            mode="markers",
            marker=dict(
                symbol="cross",
                color=list(self.mapClus3Did_color.values()),
                size=self.clus3D_df["clus3D_size"]*2,
            ),
            hovertemplate="clus3D_x=%{x}<br>clus3D_y=%{y}<br>clus3D_z=%{z}<br>clus3D_size=%{marker.size}<extra></extra>",
            )
        )
        return self

    def add2DClusters(self):
        showLegend = True
        for clus3D_id, grouped_df in self.clus2D_df.groupby("clus3D_id"):
            self.fig.add_trace(go.Scatter3d(
                mode="markers",
                legendgroup="cluster2D",
                legendgrouptitle_text="2D clusters",
                name=f"Trackster nb {clus3D_id}",
                x=grouped_df["clus2D_x"], y=grouped_df["clus2D_y"], z=grouped_df["clus2D_z"], 
                marker=dict(
                    symbol="circle",
                    color=self.clus2D_df.index.get_level_values(level="clus2D_id").map(self.mapClus2Did_color),
                    size=grouped_df["clus2D_size"],
                    line=dict(
                        color=self.mapClus3Did_color[clus3D_id],
                        width=3.
                    )
                ),
                customdata=np.dstack((grouped_df.clus2D_energy, grouped_df.clus2D_rho, grouped_df.clus2D_delta,
                    grouped_df.clus2D_pointType.map({0:"Follower", 1:"Seed", 2:"Outlier"}),
                    grouped_df.clus2D_layer))[0],
                #hovertemplate="clus2D_x=%{x}<br>clus2D_y=%{y}<br>clus2D_z=%{z}<br>clus2D_size=%{marker.size}<extra></extra>",
                hovertemplate=(
                    "2D cluster : %{customdata[3]}<br>"
                    "Layer : %{customdata[4]}<br>"
                    "Energy: %{customdata[0]:.2g} GeV<br>Rho: %{customdata[1]:.2g} GeV<br>"
                    "Delta: %{customdata[2]:.2g} cm<br>"
                    "Size: %{marker.size}"
                )
            ))

            for row in grouped_df.dropna().itertuples(index=False, name="Cluster2D"):
                self.fig.add_traces(makeArrow3D(
                    row.clus2D_x, row.clus2D_x_ofNearestHigher, row.clus2D_y, row.clus2D_y_ofNearestHigher, row.clus2D_z, row.clus2D_z_ofNearestHigher,
                    dictLine=dict(
                        name="Cluster 2D chain",
                        legendgroup="clus2D_chain",
                        showlegend=showLegend,
                        line_width=max(1, math.log(row.clus2D_cumulativeEnergy/0.1)), #line width in pixels
                    ), dictCone=dict(
                        legendgroup="clus2D_chain"
                    ),
                    color=self.mapClus3Did_color[clus3D_id],
                    )
                )

                showLegend = False
        return self
        
    def addRechits(self):
        showLegend = True
        for index, grouped_df in self.rechits_df.groupby(level=["clus3D_id", "clus2D_id"]):
            self.fig.add_trace(go.Scatter3d(
                mode="markers",
                legendgroup="rechits",
                legendgrouptitle_text="Rechits",
                name=f"2D cluster nb {index[1]}",
                x=grouped_df["rechits_x"], y=grouped_df["rechits_y"], z=grouped_df["rechits_z"], 
                marker=dict(
                    symbol="circle",
                    color=self.mapClus2Did_color[index[1]],
                    size=np.log(grouped_df["rechits_energy"]/0.0002).clip(lower=1),
                ),
                customdata=np.dstack((grouped_df.rechits_energy, grouped_df.rechits_rho, grouped_df.rechits_delta,
                    grouped_df.rechits_pointType.map({0:"Follower", 1:"Seed", 2:"Outlier"})))[0],
                #hovertemplate="clus2D_x=%{x}<br>clus2D_y=%{y}<br>clus2D_z=%{z}<br>clus2D_size=%{marker.size}<extra></extra>",
                hovertemplate=(
                    "Rechit : %{customdata[3]}<br>"
                    "Energy: %{customdata[0]:.2g} GeV<br>Rho: %{customdata[1]:.2g} GeV<br>"
                    "Delta: %{customdata[2]:.2g} cm"
                )
            ))
            for row in grouped_df.dropna().itertuples(index=False):
                self.fig.add_traces(makeArrow3D(
                    row.rechits_x, row.rechits_x_ofNearestHigher, row.rechits_y, row.rechits_y_ofNearestHigher, row.rechits_z, row.rechits_z_ofNearestHigher,
                    dictLine=dict(
                        name="Rechits chain",
                        legendgroup="rechits_chain",
                        showlegend=showLegend,
                        line_width=max(1, math.log(row.rechits_cumulativeEnergy/0.01)), #line width in pixels
                    ), dictCone=dict(
                        legendgroup="rechits_chain"
                    ),
                    color=self.mapClus2Did_color[index[1]],
                    )
                )
                showLegend = False
        return self
